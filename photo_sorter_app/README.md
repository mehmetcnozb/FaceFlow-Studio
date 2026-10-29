# 📸 Fotoğraf Ayıklayıcı — Yüz Tanıma ile Akıllı Sınıflandırma

Google Drive'daki veya bilgisayarınızdaki fotoğraflar arasından, seçtiğiniz bir kişinin
yüzüne sahip fotoğrafları otomatik olarak bulup ayrı bir klasöre kopyalar.

**Yüz Tanıma Motoru:** InsightFace (ArcFace) + ONNX Runtime **GPU (CUDA / TensorRT)**

---

## 🚀 Hızlı Başlangıç

### 1. Python Kurulumu

Python 3.9 veya üzeri gereklidir. [python.org](https://www.python.org/downloads/) adresinden indirin.

Kurulum sırasında **"Add Python to PATH"** seçeneğini işaretlemeyi unutmayın!

### 2. Kütüphaneleri Yükleme

Komut istemini (cmd) veya PowerShell'i açın, proje klasörüne gidin ve şu komutu çalıştırın:

```bash
cd photo_sorter_app
pip install -r requirements.txt
```

> **Not:** `onnxruntime-gpu` paketi NVIDIA CUDA destekli GPU gerektirir.
> GPU yoksa otomatik olarak CPU moduna düşer.

> **Not:** İlk çalıştırmada InsightFace modeli (~300 MB) otomatik olarak
> indirilir. Bu sadece bir kez olur.

### 3. Uygulamayı Çalıştırma

```bash
python main.py
```

---

## ⚡ GPU Hızlandırma

Uygulama NVIDIA GPU'nuzu otomatik olarak algılar ve kullanır:

| Provider | Hız | Açıklama |
|---|---|---|
| **TensorrtExecutionProvider** | ⚡⚡⚡ En hızlı | NVIDIA TensorRT optimizasyonu |
| **CUDAExecutionProvider** | ⚡⚡ Hızlı | NVIDIA CUDA ile GPU hızlandırma |
| **CPUExecutionProvider** | ⚡ Normal | GPU yoksa otomatik CPU modu |

### GPU Gereksinimleri
- NVIDIA GPU (GTX 1060+ / RTX serisi önerilir)
- CUDA Toolkit 11.8 veya 12.x
- cuDNN 8.x veya 9.x
- `onnxruntime-gpu` paketi (`pip install onnxruntime-gpu`)

---

## 🔑 Google Cloud credentials.json Nasıl Alınır?

Google Drive klasörlerindeki fotoğraflara erişmek için bir `credentials.json` dosyasına
ihtiyacınız var. Aşağıdaki adımları takip edin:

### Adım 1: Google Cloud Console'a Gidin

1. Tarayıcınızda şu adrese gidin: [https://console.cloud.google.com](https://console.cloud.google.com)
2. Google hesabınızla giriş yapın

### Adım 2: Yeni Proje Oluşturun

1. Sayfanın en üstünde, "Google Cloud" yazısının yanındaki **proje seçici** menüsüne tıklayın
2. Açılan pencerede **"YENİ PROJE"** (veya "NEW PROJECT") butonuna tıklayın
3. Proje adı olarak `FotoAyiklayici` yazın (istediğiniz adı verebilirsiniz)
4. **"OLUŞTUR"** butonuna tıklayın
5. Proje oluşturulduktan sonra o projeyi seçin

### Adım 3: Google Drive API'yi Etkinleştirin

1. Sol menüden **"API'ler ve Hizmetler"** → **"Kitaplık"** (Library) seçeneğine gidin
2. Arama çubuğuna **"Google Drive API"** yazın
3. Çıkan sonuçtan **"Google Drive API"** tıklayın
4. **"ETKİNLEŞTİR"** (ENABLE) butonuna tıklayın

### Adım 4: OAuth İzin Ekranını Ayarlayın

1. Sol menüden **"API'ler ve Hizmetler"** → **"OAuth izin ekranı"** (OAuth consent screen) seçeneğine gidin
2. Kullanıcı türü olarak **"Harici"** (External) seçin → **"OLUŞTUR"**
3. Açılan formda:
   - **Uygulama adı**: `Fotoğraf Ayıklayıcı` yazın
   - **Kullanıcı destek e-postası**: Kendi e-postanızı seçin
   - **Geliştirici iletişim bilgileri**: Kendi e-postanızı yazın
4. **"KAYDET VE DEVAM ET"** butonuna tıklayın
5. **"Kapsamlar"** (Scopes) sayfasında değişiklik yapmadan **"KAYDET VE DEVAM ET"**
6. **"Test kullanıcıları"** sayfasında:
   - **"ADD USERS"** butonuna tıklayın
   - Kendi Gmail adresinizi ekleyin
   - **"KAYDET VE DEVAM ET"**

### Adım 5: OAuth Kimlik Bilgileri Oluşturun

1. Sol menüden **"API'ler ve Hizmetler"** → **"Kimlik bilgileri"** (Credentials) seçeneğine gidin
2. Üstteki **"+ KİMLİK BİLGİLERİ OLUŞTUR"** butonuna tıklayın
3. Açılan menüden **"OAuth istemci kimliği"** (OAuth client ID) seçin
4. **Uygulama türü** olarak **"Masaüstü uygulaması"** (Desktop app) seçin
5. Ad olarak `FotoAyiklayici` yazın
6. **"OLUŞTUR"** butonuna tıklayın

### Adım 6: credentials.json Dosyasını İndirin

1. Kimlik bilgileri oluşturulduktan sonra açılan pencerede **"JSON İNDİR"** butonuna tıklayın
2. İndirilen dosyanın adını **`credentials.json`** olarak değiştirin
3. Bu dosyayı **`photo_sorter_app`** klasörüne kopyalayın (main.py dosyasının yanına)

> ⚠️ **ÖNEMLİ:** `credentials.json` dosyasını kimseyle paylaşmayın!
> Bu dosya sizin Google hesabınıza erişim sağlar.

### Adım 7: Uygulamayı Kullanın

1. `python main.py` ile uygulamayı başlatın
2. **"🔑 Giriş Yap"** butonuna tıklayın
3. Tarayıcınızda Google giriş sayfası açılacak
4. Google hesabınızla giriş yapın ve izin verin
5. "Bu uygulama doğrulanmadı" uyarısı çıkarsa:
   - **"Gelişmiş"** (Advanced) bağlantısına tıklayın
   - **"FotoAyiklayici'ye git (güvenli değil)"** bağlantısına tıklayın
   - İzinleri onaylayın
6. Giriş başarılı olduğunda uygulama otomatik olarak bağlanacak

---

## 📖 Kullanım Kılavuzu

### Google Drive'dan Fotoğraf Ayıklama

1. **Referans fotoğraf seçin:** Sol paneldeki "Referans Fotoğraf Seç" butonuyla
   aradığınız kişinin net bir yüz fotoğrafını seçin
2. **Google Drive'a bağlanın:** "🔑 Giriş Yap" butonuyla hesabınıza bağlanın
3. **Drive linkini yapıştırın:** Google Drive'daki klasörün paylaşım linkini
   "Google Drive Klasör Linki" alanına yapıştırın
4. **Çıktı klasörü seçin:** Eşleşen fotoğrafların kaydedileceği klasörü belirleyin
5. **Hassasiyeti ayarlayın:** Sol paneldeki slider ile benzerlik eşiğini ayarlayın
   - **0.20-0.30:** Esnek — daha fazla eşleşme bulur ama hatalı eşleşme riski artar
   - **0.35-0.45:** Dengeli — önerilen ayar
   - **0.50-0.70:** Katı — sadece çok benzer yüzler eşleşir
6. **Alt klasör tarama:** "Alt klasörleri de tara" seçeneği ile iç içe klasörleri de tarayabilirsiniz
7. **Başlatın:** "▶️ Ayıklamayı Başlat" butonuna tıklayın

### Yerel Klasörden Fotoğraf Ayıklama

1. Kaynak seçiminde **"💻 Yerel Klasör"** sekmesine geçin
2. Bilgisayarınızdaki fotoğraf klasörünü seçin
3. Diğer adımlar aynı

---

## ❓ Sık Sorulan Sorular

**S: İlk çalıştırmada çok uzun sürüyor, normal mi?**

C: Evet! İlk referans fotoğraf seçiminde InsightFace modeli (~300 MB) otomatik olarak
indirilir. Bu sadece bir kez olur, sonraki kullanımlarda çok daha hızlıdır.

**S: GPU kullanılıyor mu nasıl anlarım?**

C: Uygulama başlatıldığında sağ üst köşede GPU durumu gösterilir:
- 🟢 **GPU Aktif: CUDAExecutionProvider** — GPU kullanılıyor
- 🟡 **CPU Modu** — GPU bulunamadı, CPU ile çalışıyor

**S: "credentials.json bulunamadı" hatası alıyorum?**

C: `credentials.json` dosyasının `photo_sorter_app` klasörü içinde,
`main.py` dosyasıyla aynı dizinde olduğundan emin olun.

**S: Google'da "Bu uygulama doğrulanmadı" uyarısı çıkıyor?**

C: Bu normaldir çünkü uygulamanız henüz Google tarafından doğrulanmamıştır.
"Gelişmiş" → "... güvenli değil" seçeneklerini tıklayarak devam edebilirsiniz.
Bu sadece sizin kendi hesabınız olduğu için güvenlidir.

**S: Eşleşme bulunamıyor, ne yapmalıyım?**

C: Hassasiyet (threshold) değerini düşürerek daha esnek eşleşme sağlayabilirsiniz.
Ayrıca referans fotoğrafın net ve yüzün açıkça görünür olduğundan emin olun.

**S: Google Drive kullanmadan sadece yerel klasörle çalışabilir miyim?**

C: Evet! "💻 Yerel Klasör" sekmesine geçip bilgisayarınızdaki bir klasörü seçebilirsiniz.
Bu durumda `credentials.json` dosyasına gerek yoktur.

**S: Alt klasörlerdeki fotoğrafları da tarıyor mu?**

C: Evet! "Alt klasörleri de tara (recursive)" seçeneği varsayılan olarak açıktır.
Hem Drive hem yerel klasör modunda alt klasörleri de tarar.

---

## 📁 Dosya Yapısı

```
photo_sorter_app/
├── main.py              # Ana uygulama kodu (GPU destekli)
├── requirements.txt     # Python kütüphaneleri
├── credentials.json     # Google API kimlik bilgileri (siz ekleyeceksiniz)
├── token.json           # Otomatik oluşturulur (giriş sonrası)
└── README.md            # Bu dosya
```

---

## 🛠️ Teknik Detaylar

- **Yüz Tanıma:** InsightFace (ArcFace modeli) + ONNX Runtime GPU
- **GPU Desteği:** CUDA / TensorRT / CPU otomatik seçim
- **Benzerlik Ölçümü:** Cosine Similarity (normalize edilmiş embedding'ler)
- **GUI:** CustomTkinter (modern dark theme)
- **Google Drive:** OAuth 2.0 ile Drive API v3
- **Bellek Optimizasyonu:** Drive dosyaları doğrudan bellekte işlenir (disk I/O minimize)
- **Recursive Tarama:** Alt klasörlerin otomatik taranması (Drive + yerel)

---

## 📝 Lisans

Bu proje kişisel kullanım için oluşturulmuştur.
