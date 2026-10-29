"""
📸 Fotoğraf Ayıklayıcı - Yüz Tanıma ve Google Drive Entegrasyonu
================================================================
Yüz tanıma motoru: InsightFace (ArcFace) + ONNX Runtime (GPU CUDA / TensorRT)
GPU Hızlandırma: NVIDIA RTX serisi destekli
"""

import os
import sys
import io
import re
import shutil
import threading
import tempfile
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from file_safety import copy_unique

import customtkinter as ctk
from PIL import Image
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from tkinter import filedialog, messagebox, StringVar

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ─── Sabitler ─────────────────────────────────────────────────────────────────
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
APP_DIR = Path(__file__).parent
MAX_DOWNLOAD_WORKERS = 8  # Paralel indirme thread sayısı


# ═════════════════════════════════════════════════════════════════════════════
#  Google Drive Yöneticisi
# ═════════════════════════════════════════════════════════════════════════════
class GoogleDriveManager:
    def __init__(self):
        self.credentials_path = APP_DIR / 'credentials.json'
        self.token_path = APP_DIR / 'token.json'
        self.service = None
        self.creds = None
        self.authenticated = False

    def authenticate(self):
        """Google Drive OAuth2 kimlik doğrulaması yapar."""
        if self.token_path.exists():
            self.creds = Credentials.from_authorized_user_file(
                str(self.token_path), SCOPES
            )

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"'{self.credentials_path}' bulunamadı!\n\n"
                        "Google Cloud Console'dan credentials.json dosyasını indirip\n"
                        f"'{APP_DIR}' klasörüne koymanız gerekiyor.\n\n"
                        "Detaylı bilgi için README.md dosyasını okuyun."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                self.creds = flow.run_local_server(port=0)

            with open(self.token_path, 'w') as token_file:
                token_file.write(self.creds.to_json())

        self.service = build('drive', 'v3', credentials=self.creds)
        self.authenticated = True
        return True

    @staticmethod
    def extract_folder_id(url: str) -> str | None:
        """Drive paylaşım linkinden klasör ID'sini çıkarır."""
        patterns = [
            r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)',
            r'drive\.google\.com/drive/u/\d+/folders/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        # Direkt folder ID girilmişse
        if re.match(r'^[a-zA-Z0-9_-]{10,}$', url.strip()):
            return url.strip()
        return None

    def list_images_in_folder(self, folder_id: str, recursive: bool = True) -> list[dict]:
        """
        Belirtilen Drive klasöründeki tüm resimleri listeler.
        recursive=True ise alt klasörleri de tarar.
        """
        if not self.service:
            raise RuntimeError("Önce Google Drive'a giriş yapın!")

        images = []
        folders_to_scan = [folder_id]

        while folders_to_scan:
            current_folder = folders_to_scan.pop(0)
            page_token = None

            # Resimleri bul
            image_query = (
                f"'{current_folder}' in parents and "
                f"mimeType contains 'image/' and "
                f"trashed = false"
            )
            while True:
                results = self.service.files().list(
                    q=image_query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, size)',
                    pageToken=page_token,
                    pageSize=1000
                ).execute()
                images.extend(results.get('files', []))
                page_token = results.get('nextPageToken')
                if not page_token:
                    break

            # Alt klasörleri bul (recursive tarama)
            if recursive:
                subfolder_query = (
                    f"'{current_folder}' in parents and "
                    f"mimeType = 'application/vnd.google-apps.folder' and "
                    f"trashed = false"
                )
                page_token = None
                while True:
                    results = self.service.files().list(
                        q=subfolder_query,
                        spaces='drive',
                        fields='nextPageToken, files(id, name)',
                        pageToken=page_token,
                        pageSize=1000
                    ).execute()
                    for folder in results.get('files', []):
                        folders_to_scan.append(folder['id'])
                    page_token = results.get('nextPageToken')
                    if not page_token:
                        break

        return images

    def download_file(self, file_id: str, dest_path: str) -> str:
        """Drive'dan dosyayı indirir."""
        if not self.service:
            raise RuntimeError("Önce Google Drive'a giriş yapın!")

        request = self.service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        buffer.seek(0)
        with open(dest_path, 'wb') as f:
            f.write(buffer.read())

        return dest_path

    def download_file_to_memory(self, file_id: str) -> bytes:
        """Drive'dan dosyayı doğrudan belleğe indirir (disk I/O'suz)."""
        if not self.service:
            raise RuntimeError("Önce Google Drive'a giriş yapın!")

        request = self.service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        buffer.seek(0)
        return buffer.read()


# ═════════════════════════════════════════════════════════════════════════════
#  Yüz Tanıma Motoru — GPU Hızlandırmalı (CUDA / TensorRT)
# ═════════════════════════════════════════════════════════════════════════════
class FaceRecognizer:
    def __init__(self, log_callback=None):
        self.reference_encoding = None
        self.reference_loaded = False
        self._log = log_callback or (lambda msg: None)

        # Hangi provider gerçekten çalışıyor test et
        working_provider = self._detect_best_provider()
        self.active_provider = working_provider

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=[working_provider]
        )

        if working_provider == 'CPUExecutionProvider':
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        else:
            self.app.prepare(ctx_id=0, det_size=(640, 640))

        self._log(f"✅ Yüz tanıma motoru hazır — Aktif: {self.active_provider}")

    def _detect_best_provider(self) -> str:
        """GPU provider'ları test eder, çalışanı döndürür."""
        import onnxruntime as ort

        available = ort.get_available_providers()
        self._log(f"🔍 Mevcut ONNX Providerlar: {available}")

        # Sırayla dene: CUDA > CPU (TensorRT de CUDA gerektirir)
        for provider in ['CUDAExecutionProvider', 'CPUExecutionProvider']:
            if provider not in available:
                continue
            if provider == 'CPUExecutionProvider':
                self._log("🟡 GPU kullanılamadı, CPU modu ile devam ediliyor")
                return provider
            try:
                # Küçük bir dummy model ile test et
                import numpy as _np
                dummy_input = _np.random.randn(1, 3, 64, 64).astype(_np.float32)
                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 3  # Uyarıları bastır
                # Basit bir test: provider'ı yüklemeye çalış
                ort.InferenceSession(
                    self._create_dummy_onnx(),
                    sess_options=sess_opts,
                    providers=[provider]
                )
                if provider == 'CUDAExecutionProvider':
                    self._log("🚀 GPU modu etkin: CUDA ile çalışılacak")
                return provider
            except Exception as e:
                self._log(f"⚠️ {provider} test başarısız: {str(e)[:80]}")
                continue

        return 'CPUExecutionProvider'

    @staticmethod
    def _create_dummy_onnx() -> bytes:
        """GPU testi için minimal ONNX modeli oluşturur."""
        import onnx
        from onnx import helper, TensorProto
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'test', [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        return model.SerializeToString()

    def load_reference(self, image_path: str) -> int:
        """Referans fotoğraftan yüz embedding'i çıkarır."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Fotoğraf okunamadı.")

        faces = self.app.get(img)
        if not faces:
            raise ValueError("Referans fotoğrafta yüz bulunamadı!")

        # En büyük yüzü bul
        faces = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True
        )
        self.reference_encoding = faces[0].embedding
        # Normalize et (cosine similarity hızlanır)
        self.reference_encoding = self.reference_encoding / np.linalg.norm(self.reference_encoding)
        self.reference_loaded = True
        return len(faces)

    def check_match(self, image_path: str, tolerance: float = 0.5) -> tuple[bool, float]:
        """Verilen fotoğraftaki yüzleri referansla karşılaştırır."""
        if not self.reference_loaded:
            return False, 0.0

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Fotoğraf okunamadı.")

            faces = self.app.get(img)
            if not faces:
                return False, 0.0

            for face in faces:
                emb = face.embedding / np.linalg.norm(face.embedding)
                sim = float(np.dot(self.reference_encoding, emb))

                if sim > tolerance:
                    confidence = round(sim * 100, 1)
                    return True, confidence

            return False, 0.0

        except Exception as exc:
            raise RuntimeError(f"Fotoğraf analizi başarısız: {exc}") from exc

    def check_match_from_bytes(self, image_bytes: bytes, tolerance: float = 0.5) -> tuple[bool, float]:
        """Bellekteki görüntü verisini referansla karşılaştırır (disk I/O gerektirmez)."""
        if not self.reference_loaded:
            return False, 0.0

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Fotoğraf okunamadı.")

            faces = self.app.get(img)
            if not faces:
                return False, 0.0

            for face in faces:
                emb = face.embedding / np.linalg.norm(face.embedding)
                sim = float(np.dot(self.reference_encoding, emb))

                if sim > tolerance:
                    confidence = round(sim * 100, 1)
                    return True, confidence

            return False, 0.0

        except Exception as exc:
            raise RuntimeError(f"Fotoğraf analizi başarısız: {exc}") from exc


# ═════════════════════════════════════════════════════════════════════════════
#  Ana Uygulama (CustomTkinter GUI)
# ═════════════════════════════════════════════════════════════════════════════
class PhotoSorterApp(ctk.CTk):
    C_BG = "#0f0f1a"
    C_CARD = "#1a1a2e"
    C_INPUT = "#16213e"
    C_ACCENT = "#e94560"
    C_ACCENT_H = "#ff6b81"
    C_SUCCESS = "#00d2d3"
    C_SUCCESS_H = "#00e6e7"
    C_TEXT = "#eaeaea"
    C_TEXT_DIM = "#8d8d9b"
    C_BORDER = "#2a2a4a"
    C_GPU = "#76ff03"

    def __init__(self):
        super().__init__()
        self.title("📸 Fotoğraf Ayıklayıcı — Yüz Tanıma (GPU Hızlandırmalı)")
        self.geometry("1050x780")
        self.minsize(950, 700)
        self.configure(fg_color=self.C_BG)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.reference_path = None
        self.output_dir = None
        self.drive_manager = GoogleDriveManager()
        self.face_recognizer = None  # UI yüklendikten sonra başlatacağız
        self.is_processing = False
        self.cancel_requested = False
        self._photo_image = None

        self.drive_url_var = StringVar()
        self.tolerance_var = ctk.DoubleVar(value=0.45)
        self.local_folder_var = StringVar()
        self.source_mode = "drive"
        self.recursive_var = ctk.BooleanVar(value=True)

        self.stat_scanned = 0
        self.stat_matched = 0
        self.stat_errors = 0
        self._start_time = None

        self.ui_events = Queue()
        self._build_ui()
        self.after(50, self._drain_ui_events)
        # Yüz tanıma motorunu arka planda başlat (arayüzün donmasını engeller)
        threading.Thread(target=self._init_face_recognizer, daemon=True).start()

    def _init_face_recognizer(self):
        self._log_safe("⏳ Yüz tanıma motoru hazırlanıyor (GPU algılanıyor)...")
        try:
            recognizer = FaceRecognizer(log_callback=self._log_safe)
        except Exception as e:
            self._log_safe(f"⚠️ GPU başlatılamadı, CPU moduna geçiliyor: {e}")
            try:
                # GPU başarısız olursa CPU'ya düş
                recognizer = FaceRecognizer.__new__(FaceRecognizer)
                recognizer.reference_encoding = None
                recognizer.reference_loaded = False
                recognizer._log = self._log_safe
                recognizer.app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                recognizer.app.prepare(ctx_id=-1, det_size=(640, 640))
                recognizer.active_provider = 'CPUExecutionProvider'
                self._log_safe("✅ Yüz tanıma motoru hazır (CPU modu)")
            except Exception as e2:
                recognizer = None
                self._log_safe(f"❌ Motor başlatılamadı: {e2}")
                self._post_ui(lambda msg=str(e2): messagebox.showerror("Motor yüklenemedi", msg))

        if recognizer is not None:
            def ready():
                self.face_recognizer = recognizer
                self._log("✅ Yüz tanıma motoru hazır.")
            self._post_ui(ready)

    def _post_ui(self, callback):
        self.ui_events.put(callback)

    def _drain_ui_events(self):
        for _ in range(100):
            try:
                callback = self.ui_events.get_nowait()
            except Empty:
                break
            try:
                callback()
            except Exception:
                self.report_callback_exception(*sys.exc_info())
        self.after(50, self._drain_ui_events)

    def _log_safe(self, message: str):
        """Thread-safe log fonksiyonu (herhangi bir thread'den çağrılabilir)."""
        self._post_ui(lambda: self._log(message))

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_main_area()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=self.C_CARD, corner_radius=0, height=65)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        ctk.CTkLabel(
            header, text="📸  Fotoğraf Ayıklayıcı",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=self.C_ACCENT,
        ).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        # GPU durum göstergesi
        self.gpu_label = ctk.CTkLabel(
            header, text="⏳ GPU algılanıyor...",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=self.C_TEXT_DIM
        )
        self.gpu_label.grid(row=0, column=1, padx=20, pady=18, sticky="e")

    def _build_main_area(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=15, pady=(8, 15))
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self._build_left_panel(main)
        self._build_right_panel(main)

    def _build_left_panel(self, parent):
        panel = ctk.CTkFrame(parent, width=290, fg_color=self.C_CARD, corner_radius=14)
        panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        panel.grid_propagate(False)

        ctk.CTkLabel(
            panel, text="👤  Referans Fotoğraf", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(padx=18, pady=(18, 8))

        self.photo_frame = ctk.CTkFrame(
            panel, width=240, height=240, fg_color=self.C_INPUT, corner_radius=12
        )
        self.photo_frame.pack(padx=18, pady=5)
        self.photo_frame.pack_propagate(False)

        self.photo_label = ctk.CTkLabel(
            self.photo_frame, text="Henüz fotoğraf\nseçilmedi",
            font=ctk.CTkFont(size=13), text_color=self.C_TEXT_DIM
        )
        self.photo_label.pack(expand=True)

        ctk.CTkButton(
            panel, text="📁  Referans Fotoğraf Seç", command=self._select_reference,
            fg_color=self.C_ACCENT, hover_color=self.C_ACCENT_H,
            height=40, corner_radius=10, font=ctk.CTkFont(size=13, weight="bold")
        ).pack(padx=18, pady=(10, 5), fill="x")

        sep1 = ctk.CTkFrame(panel, height=1, fg_color=self.C_BORDER)
        sep1.pack(padx=18, pady=12, fill="x")

        ctk.CTkLabel(
            panel, text="🎯  Hassasiyet (Benzerlik)", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(padx=18, anchor="w")

        self.tolerance_display = ctk.CTkLabel(
            panel, text="0.45", font=ctk.CTkFont(size=26, weight="bold"), text_color=self.C_SUCCESS
        )
        self.tolerance_display.pack(padx=18, pady=(2, 0))

        slider = ctk.CTkSlider(
            panel, from_=0.2, to=0.8, variable=self.tolerance_var,
            command=self._on_tolerance_change, progress_color=self.C_ACCENT,
            button_color=self.C_ACCENT, button_hover_color=self.C_ACCENT_H
        )
        slider.pack(padx=18, pady=(4, 2), fill="x")

        sep2 = ctk.CTkFrame(panel, height=1, fg_color=self.C_BORDER)
        sep2.pack(padx=18, pady=12, fill="x")

        stats_card = ctk.CTkFrame(panel, fg_color=self.C_INPUT, corner_radius=10)
        stats_card.pack(padx=18, pady=(0, 8), fill="x")

        self.stats_label = ctk.CTkLabel(
            stats_card,
            text="📊  İstatistikler\n──────────────\nTaranan :  0\nEşleşen :  0\nHata      :  0\nSüre      :  —",
            font=ctk.CTkFont(family="Consolas", size=12), justify="left"
        )
        self.stats_label.pack(padx=14, pady=12)

        # Hız göstergesi
        speed_card = ctk.CTkFrame(panel, fg_color=self.C_INPUT, corner_radius=10)
        speed_card.pack(padx=18, pady=(0, 18), fill="x")

        self.speed_label = ctk.CTkLabel(
            speed_card,
            text="⚡ Hız: — foto/sn",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color=self.C_GPU
        )
        self.speed_label.pack(padx=14, pady=10)

    def _build_right_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(2, weight=1)

        self._build_source_section(panel)
        self._build_action_section(panel)
        self._build_log_section(panel)

    def _build_source_section(self, parent):
        card = ctk.CTkFrame(parent, fg_color=self.C_CARD, corner_radius=14)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card, text="📂  Fotoğraf Kaynağı", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, padx=18, pady=(15, 8), sticky="w", columnspan=2)

        tab_frame = ctk.CTkFrame(card, fg_color="transparent")
        tab_frame.grid(row=1, column=0, padx=18, pady=(0, 8), sticky="w", columnspan=2)

        self.btn_tab_drive = ctk.CTkButton(
            tab_frame, text="☁️  Google Drive", command=lambda: self._switch_source("drive"),
            fg_color=self.C_ACCENT, width=155, height=36, corner_radius=8
        )
        self.btn_tab_drive.pack(side="left", padx=(0, 6))

        self.btn_tab_local = ctk.CTkButton(
            tab_frame, text="💻  Yerel Klasör", command=lambda: self._switch_source("local"),
            fg_color=self.C_BORDER, width=155, height=36, corner_radius=8
        )
        self.btn_tab_local.pack(side="left")

        # ── Google Drive Panel ──
        self.frame_drive = ctk.CTkFrame(card, fg_color="transparent")
        self.frame_drive.grid(row=2, column=0, padx=18, pady=(0, 5), sticky="ew", columnspan=2)
        self.frame_drive.grid_columnconfigure(0, weight=1)

        self.entry_drive = ctk.CTkEntry(
            self.frame_drive, textvariable=self.drive_url_var,
            placeholder_text="https://drive.google.com/drive/folders/...", height=40
        )
        self.entry_drive.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        self.btn_auth = ctk.CTkButton(
            self.frame_drive, text="🔑 Giriş Yap", command=self._authenticate_drive_thread,
            fg_color=self.C_INPUT, width=110, height=40
        )
        self.btn_auth.grid(row=1, column=1)

        # Drive durumu
        self.drive_status = ctk.CTkLabel(
            self.frame_drive, text="", font=ctk.CTkFont(size=11),
            text_color=self.C_TEXT_DIM
        )
        self.drive_status.grid(row=2, column=0, sticky="w", pady=(3, 0), columnspan=2)

        # ── Yerel Klasör Panel ──
        self.frame_local = ctk.CTkFrame(card, fg_color="transparent")
        self.frame_local.grid_columnconfigure(0, weight=1)

        self.entry_local = ctk.CTkEntry(
            self.frame_local, textvariable=self.local_folder_var,
            placeholder_text="C:\\Fotoğraflar\\...", height=40
        )
        self.entry_local.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            self.frame_local, text="📁 Gözat", command=self._select_local_folder,
            fg_color=self.C_INPUT, width=110, height=40
        ).grid(row=1, column=1)

        # ── Çıktı klasörü ──
        out_frame = ctk.CTkFrame(card, fg_color="transparent")
        out_frame.grid(row=3, column=0, padx=18, pady=(8, 5), sticky="ew", columnspan=2)
        out_frame.grid_columnconfigure(0, weight=1)

        # Alt klasör tarama seçeneği
        self.recursive_check = ctk.CTkCheckBox(
            out_frame, text="📁 Alt klasörleri de tara (recursive)",
            variable=self.recursive_var,
            font=ctk.CTkFont(size=12),
            checkbox_height=20, checkbox_width=20
        )
        self.recursive_check.grid(row=0, column=0, sticky="w", pady=(0, 6), columnspan=2)

        ctk.CTkLabel(
            out_frame, text="Çıktı Klasörü:", font=ctk.CTkFont(size=12)
        ).grid(row=1, column=0, sticky="w", pady=(0, 3), columnspan=2)

        self.entry_output = ctk.CTkEntry(out_frame, placeholder_text="Klasör seçin...", height=40)
        self.entry_output.grid(row=2, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            out_frame, text="📁 Gözat", command=self._select_output_folder,
            fg_color=self.C_INPUT, width=110, height=40
        ).grid(row=2, column=1)

    def _build_action_section(self, parent):
        card = ctk.CTkFrame(parent, fg_color=self.C_CARD, corner_radius=14)
        card.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        card.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(padx=18, pady=(15, 8), fill="x")
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        self.btn_start = ctk.CTkButton(
            btn_row, text="▶️  Ayıklamayı Başlat", command=self._start_processing,
            fg_color=self.C_SUCCESS, text_color=self.C_BG, font=ctk.CTkFont(size=15, weight="bold"), height=48
        )
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.btn_stop = ctk.CTkButton(
            btn_row, text="⏹️  Durdur", command=self._stop_processing,
            fg_color=self.C_ACCENT, font=ctk.CTkFont(size=15, weight="bold"), height=48, state="disabled"
        )
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.progress_bar = ctk.CTkProgressBar(card, progress_color=self.C_ACCENT, height=8)
        self.progress_bar.pack(padx=18, pady=(4, 4), fill="x")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(card, text="Hazır", font=ctk.CTkFont(size=12))
        self.progress_label.pack(padx=18, pady=(0, 14))

    def _build_log_section(self, parent):
        card = ctk.CTkFrame(parent, fg_color=self.C_CARD, corner_radius=14)
        card.grid(row=2, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.grid(row=0, column=0, padx=18, pady=(12, 0), sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(top, text="📋  İşlem Günlüğü", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            top, text="🗑️ Temizle", command=self._clear_log,
            fg_color=self.C_INPUT, width=80, height=28, corner_radius=6,
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=1, sticky="e")

        self.log_text = ctk.CTkTextbox(card, fg_color=self.C_INPUT, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_text.grid(row=1, column=0, padx=18, pady=(8, 18), sticky="nsew")

    # ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}]  {message}\n")
        self.log_text.see("end")

        # GPU durumunu güncelle (motor başlatıldığında)
        if self.face_recognizer and hasattr(self.face_recognizer, 'active_provider'):
            provider = self.face_recognizer.active_provider
            if 'CUDA' in provider or 'Tensorrt' in provider:
                self.gpu_label.configure(
                    text=f"🟢 GPU Aktif: {provider}",
                    text_color=self.C_GPU
                )
            else:
                self.gpu_label.configure(
                    text=f"🟡 CPU Modu",
                    text_color="#ffab00"
                )

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _update_stats(self):
        elapsed = ""
        if self._start_time:
            dt = time.time() - self._start_time
            mins, secs = divmod(int(dt), 60)
            elapsed = f"{mins}dk {secs}sn"
        else:
            elapsed = "—"

        self.stats_label.configure(
            text=f"📊  İstatistikler\n──────────────\nTaranan :  {self.stat_scanned}\nEşleşen :  {self.stat_matched}\nHata      :  {self.stat_errors}\nSüre      :  {elapsed}"
        )

        # Hız hesaplama
        if self._start_time and self.stat_scanned > 0:
            dt = time.time() - self._start_time
            if dt > 0:
                speed = self.stat_scanned / dt
                self.speed_label.configure(text=f"⚡ Hız: {speed:.1f} foto/sn")

    def _on_tolerance_change(self, value):
        self.tolerance_display.configure(text=f"{float(value):.2f}")

    def _switch_source(self, mode: str):
        self.source_mode = mode
        if mode == "drive":
            self.frame_local.grid_remove()
            self.frame_drive.grid(row=2, column=0, padx=18, pady=(0, 5), sticky="ew", columnspan=2)
            self.btn_tab_drive.configure(fg_color=self.C_ACCENT)
            self.btn_tab_local.configure(fg_color=self.C_BORDER)
        else:
            self.frame_drive.grid_remove()
            self.frame_local.grid(row=2, column=0, padx=18, pady=(0, 5), sticky="ew", columnspan=2)
            self.btn_tab_local.configure(fg_color=self.C_ACCENT)
            self.btn_tab_drive.configure(fg_color=self.C_BORDER)

    def _select_reference(self):
        if self.is_processing:
            return
        if not self.face_recognizer:
            messagebox.showwarning("Bekleyin", "Yüz tanıma motoru arka planda yükleniyor, lütfen birkaç saniye bekleyip tekrar deneyin.")
            return

        path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return

        self.reference_path = path
        self._log(f"📷 Referans fotoğraf seçildi: {Path(path).name}")

        try:
            img = Image.open(path)
            self._photo_image = ctk.CTkImage(light_image=img, dark_image=img, size=(220, 220))
            self.photo_label.configure(image=self._photo_image, text="")
        except Exception as e:
            self._log(f"⚠️ Önizleme yüklenemedi: {e}")

        try:
            count = self.face_recognizer.load_reference(path)
            self._log(f"✅ {count} yüz algılandı, en büyük yüz referans olarak ayarlandı.")
        except Exception as e:
            self._log(f"❌ Hata: {e}")
            messagebox.showerror("Hata", str(e))
            self.reference_path = None

    def _select_local_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.local_folder_var.set(path)

    def _select_output_folder(self):
        if self.is_processing:
            return
        path = filedialog.askdirectory()
        if path:
            self.output_dir = path
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, path)

    # ─── Google Drive Kimlik Doğrulama ────────────────────────────────────────

    def _authenticate_drive_thread(self):
        if self.is_processing:
            return
        self.btn_auth.configure(state="disabled", text="⏳ Bekleniyor...")
        self.drive_status.configure(text="Tarayıcıda giriş bekleniyor...", text_color="#ffab00")
        threading.Thread(target=self._authenticate_drive, daemon=True).start()

    def _authenticate_drive(self):
        try:
            self.drive_manager.authenticate()
            self._post_ui(lambda: self.btn_auth.configure(state="normal", text="✅ Bağlı", fg_color="#0d7a3e"))
            self._post_ui(lambda: self.drive_status.configure(
                text="✅ Google Drive bağlantısı başarılı!", text_color=self.C_SUCCESS
            ))
            self._post_ui(lambda: self._log("✅ Google Drive'a başarıyla bağlanıldı!"))
        except FileNotFoundError as e:
            self._post_ui(lambda msg=str(e): messagebox.showerror("credentials.json Bulunamadı", msg))
            self._post_ui(lambda: self.btn_auth.configure(state="normal", text="🔑 Giriş Yap"))
            self._post_ui(lambda: self.drive_status.configure(
                text="❌ credentials.json bulunamadı!", text_color=self.C_ACCENT
            ))
        except Exception as e:
            self._post_ui(lambda msg=str(e): messagebox.showerror("Bağlantı Hatası", msg))
            self._post_ui(lambda: self.btn_auth.configure(state="normal", text="🔑 Giriş Yap"))
            self._post_ui(lambda msg=str(e): self.drive_status.configure(
                text=f"❌ Hata: {msg[:60]}", text_color=self.C_ACCENT
            ))

    # ─── İşlem Başlatma / Durdurma ────────────────────────────────────────────

    def _start_processing(self):
        if self.is_processing:
            return
        if not self.reference_path or not self.face_recognizer or not self.face_recognizer.reference_loaded:
            messagebox.showwarning("Uyarı", "Lütfen önce bir referans fotoğraf seçin!")
            return
        if not self.entry_output.get().strip():
            messagebox.showwarning("Uyarı", "Lütfen bir çıktı klasörü seçin!")
            return

        if self.source_mode == "drive":
            url = self.drive_url_var.get().strip()
            if not url:
                messagebox.showwarning("Uyarı", "Lütfen Google Drive klasör linkini girin!")
                return
            folder_id = GoogleDriveManager.extract_folder_id(url)
            if not folder_id:
                messagebox.showerror("Hata", "Geçersiz Google Drive linki!\nDoğru format: https://drive.google.com/drive/folders/...")
                return
            if not self.drive_manager.authenticated:
                messagebox.showwarning("Uyarı", "Lütfen önce '🔑 Giriş Yap' butonuyla Google Drive'a bağlanın!")
                return
        else:
            local_path = self.local_folder_var.get().strip()
            if not local_path or not Path(local_path).is_dir():
                messagebox.showwarning("Uyarı", "Lütfen geçerli bir yerel klasör seçin!")
                return

        self.output_dir = str(Path(self.entry_output.get().strip()).resolve())
        if self.source_mode == "local" and Path(local_path).resolve() == Path(self.output_dir):
            messagebox.showwarning("Uyarı", "Kaynak ve çıktı klasörü farklı olmalıdır.")
            return
        self.run_mode = self.source_mode
        self.run_tolerance = self.tolerance_var.get()
        self.run_recursive = self.recursive_var.get()
        self.run_source = self.local_folder_var.get().strip()
        self.run_folder_id = folder_id if self.source_mode == "drive" else None
        self.is_processing = True
        self.cancel_requested = False
        self.stat_scanned = self.stat_matched = self.stat_errors = 0
        self._start_time = time.time()
        self._update_stats()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress_bar.set(0)
        threading.Thread(target=self._process_photos, daemon=True).start()

    def _stop_processing(self):
        self.cancel_requested = True
        self._log("⏹️ İptal istendi, mevcut işlem tamamlanınca durulacak...")

    def _finish_processing(self, failed=False):
        # Called only by the UI event queue, after all progress events.
        self.is_processing = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        status = "Başarısız" if failed else ("Durduruldu" if self.cancel_requested else ("Hatalarla tamamlandı" if self.stat_errors else "Tamamlandı"))
        if not failed and not self.cancel_requested:
            self.progress_bar.set(1.0)
        self.progress_label.configure(text=status)
        self._update_stats()
        self._log(f"{status} — Taranan: {self.stat_scanned} | Eşleşen: {self.stat_matched} | Hata: {self.stat_errors}")

    def _process_photos(self):
        failed = False
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.run_mode == "drive":
                self._process_from_drive(self.run_tolerance)
            else:
                self._process_from_local(self.run_tolerance)
        except Exception as exc:
            failed = True
            self.stat_errors += 1
            self._log_safe(f"❌ İşlem hatası: {exc}")
        finally:
            self._post_ui(lambda error=failed: self._finish_processing(error))

    def _save_match(self, source, name, tolerance):
        matched, confidence = self.face_recognizer.check_match(str(source), tolerance)
        if matched:
            saved = copy_unique(source, self.output_dir, name)
            self.stat_matched += 1
            self._log_safe(f"✅ EŞLEŞTİ (%{confidence}): {saved.name}")

    def _record_error(self, name, exc):
        self.stat_errors += 1
        self._log_safe(f"❌ {name}: {exc}")

    def _progress(self, index, total):
        self._post_ui(self._update_stats)
        self._post_ui(lambda p=index/total: self.progress_bar.set(p))

    def _process_from_drive(self, tolerance):
        self._log_safe("📡 Drive klasörü listeleniyor...")
        images = self.drive_manager.list_images_in_folder(self.run_folder_id, recursive=self.run_recursive)
        if not images:
            self._log_safe("Klasörde fotoğraf bulunamadı.")
        # Unique per-run directory: never remove an existing user folder.
        with tempfile.TemporaryDirectory(prefix="photo_sorter_", dir=self.output_dir) as temporary:
            for i, info in enumerate(images):
                if self.cancel_requested:
                    break
                name = info['name']
                source = Path(temporary) / 'download.image'
                self._post_ui(lambda n=name: self.progress_label.configure(text=f"İndiriliyor ve taranıyor: {n}"))
                try:
                    self.drive_manager.download_file(info['id'], str(source))
                    if self.cancel_requested:
                        break
                    self._save_match(source, name, tolerance)
                except Exception as exc:
                    self._record_error(name, exc)
                finally:
                    source.unlink(missing_ok=True)
                self.stat_scanned += 1
                self._progress(i+1, len(images))

    def _process_from_local(self, tolerance):
        source = Path(self.run_source)
        output = Path(self.output_dir).resolve()
        if self.run_recursive:
            images = []
            for root, dirs, files in os.walk(source):
                # Do not rescan an output directory nested in the source tree.
                dirs[:] = [d for d in dirs if (Path(root)/d).resolve() != output]
                images.extend(Path(root)/f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
        else:
            images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            self._log_safe("Klasörde fotoğraf bulunamadı.")
        for i, path in enumerate(images):
            if self.cancel_requested:
                break
            self._post_ui(lambda n=path.name: self.progress_label.configure(text=f"Taranıyor: {n}"))
            try:
                self._save_match(path, path.name, tolerance)
            except Exception as exc:
                self._record_error(path.name, exc)
            self.stat_scanned += 1
            self._progress(i+1, len(images))


if __name__ == "__main__":
    app = PhotoSorterApp()
    app.mainloop()