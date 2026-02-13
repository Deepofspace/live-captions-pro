# launcher.py - GitHub release auto-updater (multi-assets) for Live Captions
import os, sys, json, hashlib, tempfile, shutil, subprocess, time
import urllib.request
import tkinter as tk
from tkinter import messagebox

# ===================== Repo settings =====================
GITHUB_OWNER = "Deepofspace"
GITHUB_REPO  = "live-captions-pro"

API_LATEST = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"

# Required assets
ASSET_EXE     = "LiveCaptionsApp.exe"
ASSET_VERSION = "version.txt"
ASSET_SHA256  = "sha256.txt"

# Optional UI assets (download if present)
OPTIONAL_ASSETS = [
    "app.ico",
    "banner_api.png",
    "banner_caption.png",
    "banner_controls.png",
    "banner_stats.png",
    "assets.json",
]

# ===================== Paths =====================
def base_dir() -> str:
    # When frozen (PyInstaller), __file__ points to temp; sys.executable is the real launcher path.
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = base_dir()
APP_DIR  = os.path.join(BASE_DIR, "app")

os.makedirs(APP_DIR, exist_ok=True)

LOG_PATH = os.path.join(BASE_DIR, "update.log")

def log(msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


# If user mistakenly placed assets next to Launcher.exe, move them into /app once.
for _name in (ASSET_EXE, ASSET_VERSION, ASSET_SHA256, *OPTIONAL_ASSETS):
    src = os.path.join(BASE_DIR, _name)
    dst = os.path.join(APP_DIR, _name)
    try:
        if os.path.exists(src) and (not os.path.exists(dst)):
            shutil.move(src, dst)
            log(f"Moved {src} -> {dst}")
    except Exception:
        pass


LOCAL_EXE = os.path.join(APP_DIR, ASSET_EXE)
LOCAL_VER = os.path.join(APP_DIR, ASSET_VERSION)
LOCAL_SHA = os.path.join(APP_DIR, ASSET_SHA256)

# ===================== Helpers =====================
def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def parse_version(v: str):
    v = (v or "").strip().lstrip("vV")
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except Exception:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def http_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "LiveCaptionsUpdater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read().decode("utf-8", errors="replace")
    return json.loads(data)

def download_to(url: str, dest_path: str, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".tmp"
    req = urllib.request.Request(url, headers={"User-Agent": "LiveCaptionsUpdater"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out)
    os.replace(tmp, dest_path)

def asset_map(release_json: dict) -> dict:
    m = {}
    for a in (release_json.get("assets") or []):
        name = a.get("name")
        url = a.get("browser_download_url")
        if name and url:
            m[name] = url
    return m

def parse_sha256_manifest(text: str) -> dict:
    """
    Supports either:
      <hash>  filename
      filename:<hash>
    Returns {filename: hash}
    """
    out = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "  " in line:
            h, fn = line.split(None, 1)
            out[fn.strip()] = h.strip()
        elif ":" in line:
            fn, h = line.split(":", 1)
            out[fn.strip()] = h.strip()
    return out

def verify_hashes(app_dir: str, sha_text: str, filenames: list) -> (bool, str):
    manifest = parse_sha256_manifest(sha_text)
    for fn in filenames:
        if fn not in manifest:
            # if not listed, skip verification (allows optional assets without hashes)
            continue
        path = os.path.join(app_dir, fn)
        if not os.path.exists(path):
            return False, f"Missing file for hash verify: {fn}"
        actual = sha256_file(path)
        expected = manifest[fn].lower()
        if actual.lower() != expected:
            return False, f"SHA256 mismatch for {fn}\nExpected: {expected}\nActual:   {actual}"
    return True, ""

# ===================== Update logic =====================

def ensure_optional_assets(amap: dict) -> None:
    """
    Best-effort download optional UI assets (banners/icon/assets.json) if missing.
    Does not raise (logs only).
    """
    for fn in OPTIONAL_ASSETS:
        dst = os.path.join(APP_DIR, fn)
        if os.path.exists(dst):
            continue
        url = amap.get(fn)
        if not url:
            continue
        try:
            download_to(url, dst, timeout=60)
            log(f"Downloaded missing optional asset: {fn}")
        except Exception as e:
            log(f"Optional asset failed ({fn}): {e}")

def do_update_if_needed(silent: bool = True) -> bool:
    try:
        os.makedirs(APP_DIR, exist_ok=True)

        local_v = read_text(LOCAL_VER) or "0.0.0"
        log(f"BASE_DIR={BASE_DIR}")
        log(f"APP_DIR={APP_DIR}")
        log(f"local_v={local_v}")

        rel = http_json(API_LATEST, timeout=20)
        amap = asset_map(rel)

        # Must have these in release
        ver_url = amap.get(ASSET_VERSION)
        exe_url = amap.get(ASSET_EXE)
        sha_url = amap.get(ASSET_SHA256)

        if not ver_url or not exe_url or not sha_url:
            msg = "Release assets missing. Need: LiveCaptionsApp.exe, version.txt, sha256.txt"
            log(msg)
            if not silent:
                messagebox.showerror("Update", msg)
            return False

        # Download remote version to temp
        with tempfile.TemporaryDirectory() as td:
            remote_ver_path = os.path.join(td, "version.txt")
            download_to(ver_url, remote_ver_path, timeout=30)
            remote_v = read_text(remote_ver_path) or "0.0.0"
            log(f"remote_v={remote_v}")

            if parse_version(remote_v) <= parse_version(local_v):
                # Up-to-date, but we may still be missing optional UI assets.
                ensure_optional_assets(amap)
                return False  # up to date

            # Download sha256 manifest first
            remote_sha_path = os.path.join(td, "sha256.txt")
            download_to(sha_url, remote_sha_path, timeout=30)
            sha_text = open(remote_sha_path, "r", encoding="utf-8", errors="replace").read()

            # Download exe
            new_exe = os.path.join(td, ASSET_EXE)
            download_to(exe_url, new_exe, timeout=120)

            # Prepare file list to verify
            to_verify = [ASSET_EXE, ASSET_VERSION] + OPTIONAL_ASSETS
            # We will write version + optional assets into APP_DIR, then verify against sha_text if hashes present.

            # Write exe and version atomically into APP_DIR
            log("Replacing core files...")
            os.makedirs(APP_DIR, exist_ok=True)

            # Write exe
            tmp_exe = os.path.join(td, "LOCAL_EXE.tmp")
            shutil.copy2(new_exe, tmp_exe)
            os.replace(tmp_exe, LOCAL_EXE)

            # Write version
            os.replace(remote_ver_path, LOCAL_VER)

            # Write sha256 itself (helps debugging)
            os.replace(remote_sha_path, LOCAL_SHA)

            # Optional assets download (best-effort)
            for fn in OPTIONAL_ASSETS:
                url = amap.get(fn)
                if not url:
                    continue
                try:
                    download_to(url, os.path.join(APP_DIR, fn), timeout=60)
                    log(f"Downloaded optional asset: {fn}")
                except Exception as e:
                    log(f"Optional asset failed ({fn}): {e}")

            ok, why = verify_hashes(APP_DIR, sha_text, to_verify)
            if not ok:
                log(f"Hash verify failed: {why}")
                if not silent:
                    messagebox.showerror("Update", why)
                # Do not rollback automatically; but signal failure
                return False

        log(f"UPDATE OK -> {read_text(LOCAL_VER)}")
        return True

    except Exception as e:
        log(f"Update error: {e}")
        if not silent:
            messagebox.showerror("Update error", str(e))
        return False

# ===================== Launch =====================
def launch_app() -> None:
    if not os.path.exists(LOCAL_EXE):
        messagebox.showerror("Missing", f"Uygulama bulunamadı.\n{os.path.relpath(LOCAL_EXE, BASE_DIR)} yok.")
        return
    try:
        subprocess.Popen([LOCAL_EXE], cwd=APP_DIR, close_fds=True)
    except Exception as e:
        messagebox.showerror("Launch error", str(e))

def main():
    root = tk.Tk()
    root.withdraw()

    updated = do_update_if_needed(silent=True)
    if updated:
        # show small confirmation including local version for sanity
        lv = read_text(LOCAL_VER) or "?"
        messagebox.showinfo("Updated", f"Yeni sürüm indirildi ve kuruldu.\n\nLocal version: {lv}")

    launch_app()
    root.destroy()

if __name__ == "__main__":
    main()