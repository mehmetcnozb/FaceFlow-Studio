# FaceFlow Studio

Yüz tanıma ile Google Drive veya yerel klasörlerdeki fotoğrafları ayıklayan Python masaüstü uygulaması.

Kaynak kodu ve ayrıntılı kurulum yönergeleri [photo_sorter_app](photo_sorter_app/README.md) klasöründedir.

```bash
cd photo_sorter_app
pip install -r requirements.txt
python main.py
```

Google Drive bağlantısı için kendi `credentials.json` dosyanızı uygulama klasörüne ekleyin. Kimlik bilgileri ve oturum tokenları depoya dahil değildir.

## Güvenilirlik güncellemesi

Aynı isimli dosyalar numaralandırılır; geçici indirmeler yalnızca o işleme ait klasörde tutulur. Hatalar ve iptal durumu ayrı gösterilir, arayüz güncellemeleri ana iş parçacığındaki olay kuyruğundan yürütülür.

Testler: `python -m unittest discover -s photo_sorter_app -p "test_file_safety.py"` ve `python -m unittest discover -s photo_sorter_app -p "test_regressions.py"`. Gerçek Google Drive ve yüz modeliyle uçtan uca doğrulama henüz yapılmamıştır.
