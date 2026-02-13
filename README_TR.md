# Live Captions Pro (OBS Dark UI)

Bu klasör, projeyi **tek tıkla build** etmek ve GitHub Releases üzerinden **auto-update** çalıştırmak için minimum gerekli dosyaları içerir.

## Klasör yapısı
- `live_caption_ui.py` → Ana uygulama (STT + DeepL + UI)
- `launcher.py` → Güncelleyici/başlatıcı (Release assets indirir, `app/` içine kurar)
- `build.ps1` → EXE build (PyInstaller)
- `requirements.txt` → Python bağımlılıkları
- `version.txt` → Uygulama sürümü (başlıkta görünür)
- `assets.json` → Launcher'ın indireceği opsiyonel görseller listesi
- `.gitignore` → Git ignore önerisi

## İlk kurulum (Geliştirme)
1) Python 3.10+ kur.
2) Bu klasörde terminal aç:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   python live_caption_ui.py
   ```

## Build (EXE)
PowerShell için:
```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Çıktı: `dist/LiveCaptionsApp.exe` ve (build scriptin yapısına göre) `dist/Launcher.exe`.

## GitHub Release Assets (önerilen)
Release'e şu dosyaları koy:
- `LiveCaptionsApp.exe`
- `version.txt`
- `sha256.txt`
- `assets.json`
- `app.ico`
- `banner_api.png`
- `banner_caption.png`
- `banner_controls.png`
- `banner_stats.png`

> `sha256.txt` içinde hem `LiveCaptionsApp.exe` hem `version.txt` hash'i olmalı.

## Test
Test klasörü:
- `Launcher.exe`
- (boş da olabilir) `app/`

Launcher:
- `app/` yoksa oluşturur
- EXE yoksa otomatik indirir ve çalıştırır
