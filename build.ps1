# build.ps1 - Build App + Launcher and prepare dist/ for GitHub Release
$ErrorActionPreference = "Stop"

$AppName = "LiveCaptionsApp"
$LauncherName = "Launcher"

Write-Host "=== Live Captions build started ==="

# Resolve python
function Get-PythonCmd {
    if (Test-Path ".\.venv\Scripts\python.exe") { return ".\.venv\Scripts\python.exe" }
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    throw "Python bulunamadı. Python kur veya .venv oluştur."
}

$PY = Get-PythonCmd

# Ensure venv exists
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[INFO] .venv yok -> olusturuluyor..."
    & python -m venv .venv
    $PY = ".\.venv\Scripts\python.exe"
}

Write-Host "[INFO] Python: $PY"

# Install deps
Write-Host "[INFO] Installing requirements..."
& $PY -m pip install --upgrade pip | Out-Null
if (Test-Path ".\requirements.txt") {
    & $PY -m pip install -r .\requirements.txt | Out-Null
}
& $PY -m pip install pyinstaller | Out-Null

# Clean
if (Test-Path ".\build") { Remove-Item ".\build" -Recurse -Force }
if (Test-Path ".\dist")  { Remove-Item ".\dist"  -Recurse -Force }
Get-ChildItem -Filter "*.spec" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# Build App
Write-Host "[INFO] Building $AppName.exe ..."
$iconArg = @()
if (Test-Path ".\app.ico") { $iconArg = @("--icon", ".\app.ico") }

& $PY -m PyInstaller --onefile --noconsole --name $AppName @iconArg ".\live_caption_ui.py"

# Build Launcher
Write-Host "[INFO] Building $LauncherName.exe ..."
& $PY -m PyInstaller --onefile --noconsole --name $LauncherName ".\launcher.py"

# Prepare dist (PyInstaller already put exe in dist/)
$dist = ".\dist"
if (-not (Test-Path $dist)) { throw "dist klasörü oluşmadı." }

# Copy version + assets into dist for Release upload
Write-Host "[INFO] Copying release assets to dist/ ..."

if (Test-Path ".\version.txt") { Copy-Item ".\version.txt" "$dist\version.txt" -Force } else { Write-Host "[WARN] version.txt yok" }
if (Test-Path ".\assets.json") { Copy-Item ".\assets.json" "$dist\assets.json" -Force } else { Write-Host "[WARN] assets.json yok" }

if (Test-Path ".\app.ico") { Copy-Item ".\app.ico" "$dist\app.ico" -Force } else { Write-Host "[WARN] app.ico yok" }
if (Test-Path ".\banner_api.png") { Copy-Item ".\banner_api.png" "$dist\banner_api.png" -Force } else { Write-Host "[WARN] banner_api.png yok" }
if (Test-Path ".\banner_caption.png") { Copy-Item ".\banner_caption.png" "$dist\banner_caption.png" -Force } else { Write-Host "[WARN] banner_caption.png yok" }
if (Test-Path ".\banner_controls.png") { Copy-Item ".\banner_controls.png" "$dist\banner_controls.png" -Force } else { Write-Host "[WARN] banner_controls.png yok" }
if (Test-Path ".\banner_stats.png") { Copy-Item ".\banner_stats.png" "$dist\banner_stats.png" -Force } else { Write-Host "[WARN] banner_stats.png yok" }

# Generate sha256.txt (exe + version + optional assets if present)
Write-Host "[INFO] Generating sha256.txt ..."
$shaLines = @()

function Add-HashLine($path, $name) {
    if (Test-Path $path) {
        $h = (Get-FileHash $path -Algorithm SHA256).Hash
        $script:shaLines += "$h  $name"
    }
}

Add-HashLine "$dist\$AppName.exe" "$AppName.exe"
Add-HashLine "$dist\version.txt" "version.txt"
Add-HashLine "$dist\assets.json" "assets.json"
Add-HashLine "$dist\app.ico" "app.ico"
Add-HashLine "$dist\banner_api.png" "banner_api.png"
Add-HashLine "$dist\banner_caption.png" "banner_caption.png"
Add-HashLine "$dist\banner_controls.png" "banner_controls.png"
Add-HashLine "$dist\banner_stats.png" "banner_stats.png"

$shaPath = "$dist\sha256.txt"
$shaLines -join "`n" | Set-Content -Path $shaPath -Encoding UTF8

Write-Host ""
Write-Host "=== DONE ==="
Write-Host "Release'e yuklemen gerekenler dist/ icinde:"
Write-Host " - $AppName.exe"
Write-Host " - $LauncherName.exe"
Write-Host " - version.txt"
Write-Host " - sha256.txt"
Write-Host " - assets.json"
Write-Host " - app.ico"
Write-Host " - banner_*.png"
Write-Host ""
Write-Host "Kapatmak icin Enter..."
Read-Host | Out-Null
