import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading, queue, time, os, sys, wave, re, json, base64, socket, uuid, hashlib, secrets
from io import BytesIO
from difflib import SequenceMatcher

import numpy as np
import sounddevice as sd
import requests

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ======================================================
# ALL FILES SAVED IN APPLICATION FOLDER (EXE-safe)
# ======================================================
def base_dir() -> str:
    # PyInstaller frozen exe: assets/config live next to the executable
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = base_dir()

def read_version() -> str:
    try:
        p = os.path.join(BASE_DIR, "version.txt")
        if os.path.exists(p):
            v = open(p, "r", encoding="utf-8").read().strip()
            return v if v else "0.0.0"
    except Exception:
        pass
    return "0.0.0"

APP_VERSION = read_version()

from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet, InvalidToken
# ===================== Tooltip (Hover Açıklama) =====================
class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 15
        y = self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            self.tip,
            text=self.text,
            background="#202020",
            foreground="white",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=380,
            padx=8,
            pady=6
        )
        lbl.pack()

    def _hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# ===================== Help Bubble (❓) =====================
def add_help_bubble(parent, help_text: str):
    """Adds a small ❓ icon that shows a Turkish tooltip on hover."""
    bubble = tk.Label(
        parent,
        text="❓",
        bg=parent["bg"],
        fg="#9aa0a6",
        cursor="question_arrow",
        font=("Segoe UI", 10, "bold"),
    )
    bubble.pack(side="left", padx=(8, 0))
    ToolTip(bubble, help_text or "")
    return bubble

# ===================== Pricing (estimates) =====================
OPENAI_PRICE_PER_MIN = 0.006  # rough
DEEPL_PRICE_PER_MILLION_CHAR = 20.0  # rough

# ===================== App writable folder =====================
def get_appdata_dir() -> str:
    # Kept for compatibility; we now use the application folder.
    return BASE_DIR

APPDIR = BASE_DIR
OUTPUT_DIR_DEFAULT = BASE_DIR
# ===================== Secure, machine-bound config =====================
CFG_FILE = os.path.join(BASE_DIR, "config.sec")
SALT_FILE = os.path.join(BASE_DIR, "config.salt")

def _machine_fingerprint() -> bytes:
    raw = f"{socket.gethostname()}|{uuid.getnode()}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).digest()

def _load_or_create_salt() -> bytes:
    if os.path.exists(SALT_FILE):
        try:
            s = open(SALT_FILE, "rb").read()
            if len(s) >= 16:
                return s[:16]
        except OSError:
            pass
    s = secrets.token_bytes(16)
    tmp = SALT_FILE + ".tmp"
    with open(tmp, "wb") as f:
        f.write(s)
    os.replace(tmp, SALT_FILE)
    return s

def _fernet() -> Fernet:
    salt = _load_or_create_salt()
    fp = _machine_fingerprint()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=200_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(fp))
    return Fernet(key)

def save_secure_config(obj: dict) -> None:
    f = _fernet()
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    token = f.encrypt(payload)
    tmp = CFG_FILE + ".tmp"
    with open(tmp, "wb") as w:
        w.write(token)
    os.replace(tmp, CFG_FILE)

def load_secure_config() -> dict:
    if not os.path.exists(CFG_FILE):
        return {}
    try:
        f = _fernet()
        token = open(CFG_FILE, "rb").read()
        payload = f.decrypt(token)
        return json.loads(payload.decode("utf-8"))
    except (InvalidToken, OSError, json.JSONDecodeError):
        return {}

# ===================== Path resolving =====================
def resolve_path(p: str, base_dir: str = None) -> str:
    """Resolve relative filenames into the chosen output folder (base_dir) or BASE_DIR."""
    p = (p or "").strip()
    base = base_dir or BASE_DIR
    if not p:
        return os.path.join(base, "output.txt")
    if os.path.isabs(p):
        return p
    return os.path.join(base, p)

# ===================== IO helpers =====================
def atomic_write(path: str, text: str, base_dir: str) -> None:
    path = resolve_path(path, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def append_write(path: str, text: str, base_dir: str) -> None:
    path = resolve_path(path, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def to_wav_bytes(audio_f32: np.ndarray, sr: int) -> bytes:
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    return buf.getvalue()

# ===================== HTTP backoff =====================
def _is_retryable_status(code: int) -> bool:
    return code in (429, 500, 502, 503, 504)

def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 20.0):
    wait = min(cap, base * (2 ** attempt))
    wait = wait * (0.85 + 0.3 * (time.time() % 1.0))
    time.sleep(wait)

def post_with_backoff(url: str, headers: dict, files=None, data=None, timeout: int = 30, max_retries: int = 6):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
            if r.status_code == 200:
                return r
            if _is_retryable_status(r.status_code) and attempt < max_retries:
                _sleep_backoff(attempt, base=1.0, cap=20.0)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_retries:
                _sleep_backoff(attempt, base=1.0, cap=20.0)
                continue
            raise
    raise last_exc if last_exc else RuntimeError("Request failed")

# ===================== API calls =====================
def openai_transcribe_tr(wav_bytes: bytes, api_key: str, model: str, timeout_s: int) -> str:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"model": model, "language": "tr", "response_format": "json"}
    r = post_with_backoff(url, headers=headers, files=files, data=data, timeout=timeout_s, max_retries=7)
    return (r.json().get("text") or "").strip()

def deepl_translate(text_tr: str, deepl_key: str, deepl_base: str, target: str, timeout_s: int) -> str:
    url = f"{deepl_base.rstrip('/')}/v2/translate"
    headers = {
        "Authorization": f"DeepL-Auth-Key {deepl_key.strip()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"text": text_tr, "source_lang": "TR", "target_lang": target}
    r = post_with_backoff(url, headers=headers, files=None, data=data, timeout=timeout_s, max_retries=5)
    j = r.json()
    return ((j.get("translations") or [{}])[0].get("text")) or ""

# ===================== Special Names (Fuzzy) =====================
_word_re = re.compile(r"(\w+)", re.UNICODE)

def _norm(s: str) -> str:
    return s.casefold().strip()

def parse_special_names(text: str):
    mapping = {}
    canon = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            a, b = line.split("->", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                mapping[_norm(a)] = b
        elif "=>" in line:
            a, b = line.split("=>", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                mapping[_norm(a)] = b
        else:
            canon.append(line)
    canon_norm = [_norm(x) for x in canon if x.strip()]
    return mapping, canon_norm, canon

def _best_match(token_norm: str, candidates_norm, threshold: float):
    best = None
    best_score = threshold
    for cand in candidates_norm:
        if abs(len(token_norm) - len(cand)) > 6:
            continue
        score = SequenceMatcher(None, token_norm, cand).ratio()
        if score > best_score:
            best_score = score
            best = cand
    return best

def apply_special_names(text: str, mapping: dict, canon_norm, canon_original, threshold: float) -> str:
    if not text:
        return text

    def repl_exact(m):
        tok = m.group(0)
        return mapping.get(_norm(tok), tok)

    out = _word_re.sub(repl_exact, text)

    if not canon_norm:
        return out

    canon_lookup = {n: o for n, o in zip(canon_norm, canon_original) if n}

    def repl_fuzzy(m):
        tok = m.group(0)
        n = _norm(tok)
        if len(n) < 3:
            return tok
        if n in canon_lookup:
            return canon_lookup[n]
        best = _best_match(n, canon_norm, threshold)
        if best is not None:
            return canon_lookup.get(best, tok)
        return tok

    return _word_re.sub(repl_fuzzy, out)

# ===================== Segmenter =====================
class AudioSegmenter:
    def __init__(self, energy_threshold: float, silence_to_finalize: float, min_utt_s: float, max_utt_s: float):
        self.energy_threshold = energy_threshold
        self.silence_to_finalize = silence_to_finalize
        self.min_utt_s = min_utt_s
        self.max_utt_s = max_utt_s
        self.buf = []
        self.last_voice_time = None
        self.segment_start_time = None

    def feed(self, chunk: np.ndarray, now: float):
        energy = float(np.sqrt(np.mean(chunk * chunk)) + 1e-12)
        is_voice = energy >= self.energy_threshold
        if is_voice:
            if self.segment_start_time is None:
                self.segment_start_time = now
            self.last_voice_time = now
            self.buf.append(chunk)
            return False, None
        if self.buf and self.last_voice_time is not None:
            silence_dur = now - self.last_voice_time
            seg_dur = now - (self.segment_start_time or now)
            if seg_dur >= self.max_utt_s:
                return self._finalize()
            if silence_dur >= self.silence_to_finalize and seg_dur >= self.min_utt_s:
                return self._finalize()
        return False, None

    def _finalize(self):
        audio = np.concatenate(self.buf, axis=0) if self.buf else np.zeros((0,), dtype=np.float32)
        self.buf = []
        self.last_voice_time = None
        self.segment_start_time = None
        return True, audio

# ===================== App (OBS-style sidebar UI) =====================
class LiveCaptionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Live Caption v{APP_VERSION}")
        self.geometry("1400x880")
        self.configure(bg="#181818")

        self.ui_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None

        self.spoken_seconds = 0.0
        self.deepl_chars = 0
        self.output_dir = tk.StringVar(value=BASE_DIR)
        self.openai_model = tk.StringVar(value="gpt-4o-transcribe")
        self.deepl_base = tk.StringVar(value="https://api-free.deepl.com")
        self.sample_rate = tk.IntVar(value=16000)

        self.energy_threshold = tk.DoubleVar(value=0.016)
        self.silence_finalize = tk.DoubleVar(value=1.20)
        self.min_utt = tk.DoubleVar(value=1.40)
        self.max_utt = tk.DoubleVar(value=10.0)

        self.min_seconds_between_calls = tk.DoubleVar(value=2.5)
        self.openai_timeout = tk.IntVar(value=30)
        self.deepl_timeout = tk.IntVar(value=25)

        self.write_files = tk.BooleanVar(value=True)
        self.out_tr = tk.StringVar(value="caption_tr.txt")
        self.out_en = tk.StringVar(value="caption_en.txt")
        self.out_ua = tk.StringVar(value="caption_ua.txt")

        self.write_transcript = tk.BooleanVar(value=True)
        self.transcript_path = tk.StringVar(value="session_transcript.txt")
        self.transcript_clear_on_start = tk.BooleanVar(value=True)

        self.max_chars = tk.IntVar(value=42)
        self.max_lines = tk.IntVar(value=2)

        self.special_enabled = tk.BooleanVar(value=True)
        self.special_threshold = tk.DoubleVar(value=0.86)

        self.selected_device = tk.StringVar(value="")
        self.device_map = {}

        self.active_section = None
        self.running_anim = False
        self._init_assets()
        self._build_ui()

        self._refresh_devices()
        self._load_config_into_ui()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_ui_queue)
        self.after(500, self._update_stats_labels)

        self._log(f"App folder: {BASE_DIR}")
        self._log(f"Output folder: {self.output_dir.get()}")

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar = tk.Frame(self, bg="#202020", width=230)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)


        self.main = tk.Frame(self, bg="#2b2b2b")
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(0, weight=1)

        self.controls = tk.Frame(self, bg="#111111", height=65)
        self.controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.controls.columnconfigure(0, weight=1)
        self.controls.columnconfigure(1, weight=1)
        self.controls.columnconfigure(2, weight=1)

        self.save_btn = tk.Button(self.controls, text="Save Config",
                                  bg="#3a3a3a", fg="white",
                                  command=lambda: self._save_config_now(show_popup=True))
        self.save_btn.grid(row=0, column=0, pady=15)

        tk.Label(self.controls, text="By Deepofspace",
                 bg="#111111", fg="#888888",
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=1)

        self.start_btn = tk.Button(self.controls, text="Start",
                                   bg="#1f6f3f", fg="white",
                                   width=10, command=self.start)
        self.start_btn.grid(row=0, column=2, sticky="e", padx=(0, 100))

        self.stop_btn = tk.Button(self.controls, text="Stop",
                                  bg="#8b1e1e", fg="white",
                                  width=10, command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=2, sticky="e", padx=(0, 20))

        self.sections = {}
        self.section_banners = {}
        self.sidebar_buttons = {}
        names = ["API Keys","Settings","Caption Formatting",
                 "Special Names","Output Files",
                 "Controls","Live Output","Log","Stats"]

        for name in names:
            btn = tk.Button(self.sidebar, text=name, anchor="w",
                            bg="#202020", fg="#cccccc",
                            relief="flat",
                            command=lambda n=name: self.show_section(n))
            # Optional icon for sidebar button
            _ico = self._get_icon(name)
            if _ico is not None:
                btn.configure(image=_ico, compound="left", padx=10)
            btn.pack(fill="x", padx=8, pady=4)
            self.sidebar_buttons[name] = btn

            frame = tk.Frame(self.main, bg="#2b2b2b")
            frame.grid(row=0, column=0, sticky="nsew")
            self.sections[name] = frame

        self._build_sections()
        self.show_section("API Keys")

    def show_section(self, name: str):
        for frame in self.sections.values():
            frame.grid_remove()
        self.sections[name].grid()

        for n, btn in self.sidebar_buttons.items():
            btn.configure(bg="#202020", fg="#cccccc")
        self.sidebar_buttons[name].configure(bg="#3a3a3a", fg="white")
        self.active_section = name
        try:
            self._update_banner(name)
        except Exception:
            pass


    # ====================================================
    # Assets (Icon / Banners / Sidebar Icons)
    # ====================================================
    def _init_assets(self):
        '''
        Loads optional UI assets if present in the same folder:
          - app.ico / app.png for window icon
          - assets.json for mappings (optional)
          - banner_*.png files for section banners
        This is 100% optional: if files are missing, UI still works.
        '''
        self._img_cache = {}
        self._button_icons = {}
        self._assets_cfg = {}

        # Load assets.json if exists
        try:
            cfg_path = os.path.join(BASE_DIR, "assets.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self._assets_cfg = json.load(f) or {}
        except Exception:
            self._assets_cfg = {}

        # Window icon (Windows prefers .ico)
        try:
            ico_path = os.path.join(BASE_DIR, "app.ico")
            png_path = os.path.join(BASE_DIR, "app.png")
            if os.path.exists(ico_path):
                try:
                    self.iconbitmap(ico_path)
                except Exception:
                    if os.path.exists(png_path):
                        img = self._load_image("app_png", png_path)
                        if img is not None:
                            self.iconphoto(True, img)
            elif os.path.exists(png_path):
                img = self._load_image("app_png", png_path)
                if img is not None:
                    self.iconphoto(True, img)
        except Exception:
            pass

        # Preload any icon mappings for sidebar buttons (optional)
        try:
            if isinstance(self._assets_cfg, dict):
                icons_map = self._assets_cfg.get("icons", {})
                icons_b64 = self._assets_cfg.get("icons_base64", {})
            else:
                icons_map, icons_b64 = {}, {}

            if isinstance(icons_map, dict):
                for section, relpath in icons_map.items():
                    if not relpath:
                        continue
                    full = relpath if os.path.isabs(relpath) else os.path.join(BASE_DIR, relpath)
                    img = self._load_image(f"ico::{section}", full)
                    if img is not None:
                        self._button_icons[section] = img

            if isinstance(icons_b64, dict):
                for section, b64 in icons_b64.items():
                    if not b64:
                        continue
                    img = self._load_image_from_b64(f"ico_b64::{section}", b64)
                    if img is not None:
                        self._button_icons[section] = img
        except Exception:
            pass

        # Banner mapping (optional)
        self._banner_map = {}
        try:
            if isinstance(self._assets_cfg, dict):
                banners_map = self._assets_cfg.get("banners", {})
            else:
                banners_map = {}
            if isinstance(banners_map, dict):
                self._banner_map.update(banners_map)
        except Exception:
            pass

    def _load_image(self, key: str, path: str):
        if not path or not os.path.exists(path):
            return None
        if key in self._img_cache:
            return self._img_cache[key]
        try:
            img = tk.PhotoImage(file=path)
            self._img_cache[key] = img
            return img
        except Exception:
            return None

    def _load_image_from_b64(self, key: str, b64: str):
        if not b64:
            return None
        if key in self._img_cache:
            return self._img_cache[key]
        try:
            raw = base64.b64decode(b64)
            img = tk.PhotoImage(data=base64.b64encode(raw))
            self._img_cache[key] = img
            return img
        except Exception:
            return None

    def _get_icon(self, section_name: str):
        # 1) assets.json mapping (icons / icons_base64)
        if section_name in getattr(self, "_button_icons", {}):
            return self._button_icons[section_name]

        # 2) conventional filenames next to script (optional)
        #    e.g. icon_api_keys.png, icon_settings.png, etc.
        try:
            slug = re.sub(r"[^a-z0-9]+", "_", section_name.lower()).strip("_")
            candidates = [
                os.path.join(BASE_DIR, f"icon_{slug}.png"),
                os.path.join(BASE_DIR, f"{slug}.png"),
            ]
            for c in candidates:
                img = self._load_image(f"autoico::{section_name}::{c}", c)
                if img is not None:
                    self._button_icons[section_name] = img
                    return img
        except Exception:
            pass

        return None

    def _banner_key_for_section(self, section_name: str) -> str:
        # Your repo banner names: banner_api.png, banner_caption.png, banner_controls.png, banner_stats.png
        if section_name == "API Keys":
            return "api"
        if section_name in ("Caption Formatting", "Special Names"):
            return "caption"
        if section_name == "Stats":
            return "stats"
        return "controls"

    def _update_banner(self, section_name: str):
        lbl = getattr(self, 'section_banners', {}).get(section_name)
        if lbl is None:
            return
        banner_path = None

        # assets.json override (can point to any image path)
        try:
            if isinstance(getattr(self, "_banner_map", None), dict):
                banner_path = self._banner_map.get(section_name)
                if not banner_path:
                    banner_path = self._banner_map.get(self._banner_key_for_section(section_name))
        except Exception:
            banner_path = None

        if banner_path:
            banner_path = banner_path if os.path.isabs(banner_path) else os.path.join(BASE_DIR, banner_path)
        else:
            key = self._banner_key_for_section(section_name)
            banner_path = os.path.join(BASE_DIR, f"banner_{key}.png")

        img = self._load_image(f"banner::{section_name}", banner_path)
        if img is not None:
            lbl.configure(image=img, text="", anchor="w")
            lbl.image = img  # keep reference
        else:
            lbl.configure(image="", text=f"Live Captions\\n{section_name}", anchor="w")

    def _pick_output_folder(self):
        folder = filedialog.askdirectory(initialdir=self.output_dir.get() or APPDIR)
        if folder:
            self.output_dir.set(folder)
            try:
                self._log(f"Output folder set: {folder}")
            except Exception:
                pass

    def _apply_folder_to_paths(self):
        """Set file path fields to default relative filenames (written into Output Folder)."""
        self.out_tr.set("caption_tr.txt")
        self.out_en.set("caption_en.txt")
        self.out_ua.set("caption_ua.txt")
        self.transcript_path.set("session_transcript.txt")
        try:
            self._log("Applied output folder to paths (default filenames).")
        except Exception:
            pass


    def _settings_help_map(self):
        # Keys must match the Settings 'items' labels exactly.
        return {
            "OpenAI Model":
                "Kullanılacak OpenAI transcribe modeli.\n"
                "Öneri: gpt-4o-transcribe (yüksek doğruluk).",

            "DeepL Base":
                "DeepL API adresi.\n"
                "Free: https://api-free.deepl.com\n"
                "Pro:  https://api.deepl.com",

            "Min sec between OpenAI calls":
                "OpenAI istekleri arasındaki minimum süre.\n\n"
                "0.6–1.0  → daha hızlı altyazı\n"
                "2.0+     → daha az maliyet ve 429 riski azalır.",

            "OpenAI timeout (s)":
                "OpenAI isteğinin maksimum bekleme süresi (saniye).\n"
                "İnternet dalgalanıyorsa artırabilirsin.",

            "DeepL timeout (s)":
                "DeepL isteğinin maksimum bekleme süresi (saniye).\n"
                "İnternet dalgalanıyorsa artırabilirsin.",

            "Sample Rate":
                "Mikrofon örnekleme hızı (Hz).\n"
                "16000 genelde STT için idealdir.",

            "Energy Thresh":
                "Mikrofon hassasiyet eşiği (RMS).\n\n"
                "Düşürürsen daha çabuk tetikler\n"
                "Çok düşerse arka plan gürültüsünü algılayabilir.\n"
                "Öneri: 0.010–0.016 (OBS filtrelerin iyiyse).",

            "Silence finalize (s)":
                "Bu kadar sessizlik olursa cümle 'bitti' sayılır.\n\n"
                "Düşük → hızlı ama cümle bölünebilir\n"
                "Yüksek → daha düzgün ama gecikir.",

            "Min utt (s)":
                "İşlenmesi için minimum konuşma süresi.\n"
                "0.5 FAST mod için idealdir.",

            "Max utt (s)":
                "Konuşma hiç durmasa bile bu sürede bir finalize olur.\n\n"
                "0.5 → çok hızlı günceller (daha çok istek/maliyet)\n"
                "Yüksek → daha az istek (daha ucuz).",

            "Enable English (DeepL)":
                "İngilizce çeviriyi aç/kapat.\n"
                "Kapalıysa DeepL maliyeti ve gecikme azalır.",

            "Enable Ukrainian (DeepL)":
                "Ukraynaca çeviriyi aç/kapat.\n"
                "Kapalıysa DeepL maliyeti ve gecikme azalır.",
        }

    def _build_sections(self):
        pad = {"padx": 16, "pady": 10}

        # API Keys
        f = self.sections["API Keys"]
        tk.Label(f, text="OpenAI API Key", bg="#2b2b2b", fg="white").pack(**pad)
        self.openai_key_entry = tk.Entry(f, width=70, show="•")
        self.openai_key_entry.pack(padx=16)

        tk.Label(f, text="DeepL API Key", bg="#2b2b2b", fg="white").pack(**pad)
        self.deepl_key_entry = tk.Entry(f, width=70, show="•")
        self.deepl_key_entry.pack(padx=16)

        # Settings
        f = self.sections["Settings"]
        tk.Label(f, text="Settings", bg="#2b2b2b", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12)
        row = tk.Frame(f, bg="#2b2b2b")
        row.pack(fill="x", padx=16, pady=6)
        tk.Label(row, text="Input Device", bg="#2b2b2b", fg="white").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.device_combo = ttk.Combobox(row, textvariable=self.selected_device, state="readonly", width=60)
        self.device_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(row, text="Refresh", bg="#3a3a3a", fg="white", command=self._refresh_devices).grid(row=0, column=2, padx=6, pady=6)
        row.columnconfigure(1, weight=1)

        grid = tk.Frame(f, bg="#2b2b2b")
        grid.pack(fill="x", padx=16, pady=8)
        items = [
            ("OpenAI Model", self.openai_model),
            ("DeepL Base", self.deepl_base),
            ("Min sec between OpenAI calls", self.min_seconds_between_calls),
            ("OpenAI timeout (s)", self.openai_timeout),
            ("DeepL timeout (s)", self.deepl_timeout),
            ("Sample Rate", self.sample_rate),
            ("Energy Thresh", self.energy_threshold),
            ("Silence finalize (s)", self.silence_finalize),
            ("Min utt (s)", self.min_utt),
            ("Max utt (s)", self.max_utt),
        ]
        help_map = self._settings_help_map()
        for r, (lbl, var) in enumerate(items):
            tk.Label(grid, text=lbl, bg="#2b2b2b", fg="white").grid(row=r, column=0, sticky="w", padx=6, pady=6)
            cell = tk.Frame(grid, bg="#2b2b2b")
            cell.grid(row=r, column=1, sticky="ew", padx=6, pady=6)
            e = tk.Entry(cell, textvariable=var, width=60)
            e.pack(side="left", fill="x", expand=True)
            add_help_bubble(cell, help_map.get(lbl, ""))
        grid.columnconfigure(1, weight=1)

        # Caption Formatting
        f = self.sections["Caption Formatting"]
        tk.Label(f, text="Caption Formatting (Fix long sentences)", bg="#2b2b2b", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12)
        tk.Label(f, text="NOTE: Full text is always shown & written to files. (Reference only)",
                 bg="#2b2b2b", fg="#aaaaaa").pack(pady=6)
        row = tk.Frame(f, bg="#2b2b2b")
        row.pack(pady=10)
        tk.Label(row, text="Max chars/line", bg="#2b2b2b", fg="white").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        tk.Entry(row, textvariable=self.max_chars, width=6).grid(row=0, column=1, padx=8, pady=6)
        tk.Label(row, text="Lines", bg="#2b2b2b", fg="white").grid(row=0, column=2, padx=8, pady=6, sticky="w")
        tk.Entry(row, textvariable=self.max_lines, width=4).grid(row=0, column=3, padx=8, pady=6)

        # Special Names
        f = self.sections["Special Names"]
        tk.Label(f, text="Special Names (Fuzzy Fix)", bg="#2b2b2b", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12)
        top = tk.Frame(f, bg="#2b2b2b")
        top.pack(fill="x", padx=16, pady=6)
        tk.Checkbutton(top, text="Enable", variable=self.special_enabled, bg="#2b2b2b", fg="white",
                       selectcolor="#2b2b2b", activebackground="#2b2b2b").pack(side="left")
        tk.Label(top, text="Threshold", bg="#2b2b2b", fg="white").pack(side="left", padx=(16, 6))
        tk.Entry(top, textvariable=self.special_threshold, width=6).pack(side="left")

        self.special_text = tk.Text(f, height=12, wrap="word")
        self.special_text.pack(fill="both", expand=True, padx=16, pady=10)
        self.special_text.insert("1.0", "# One per line:\n#  - CorrectName\n#  - wrong -> CorrectName\n")

        # Output Files
        f = self.sections["Output Files"]
        tk.Label(f, text="Output Files (for OBS)", bg="#2b2b2b", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12)

        tk.Label(f, text=f"Tip: Relative paths are written under: {APPDIR}", bg="#2b2b2b", fg="#aaaaaa").pack(anchor="w", padx=16)

        # Output folder picker (used as base for relative paths)
        folder_row = tk.Frame(f, bg="#2b2b2b")
        folder_row.pack(fill="x", padx=16, pady=8)
        tk.Label(folder_row, text="Output Folder", bg="#2b2b2b", fg="white").pack(side="left", padx=(0, 8))
        tk.Entry(folder_row, textvariable=self.output_dir, width=60).pack(side="left", fill="x", expand=True)
        tk.Button(folder_row, text="📂 Klasör Seç", bg="#3a3a3a", fg="white",
                  command=self._pick_output_folder).pack(side="left", padx=(10, 0))
        tk.Button(folder_row, text="Apply Folder to Paths", bg="#3a3a3a", fg="white",
                  command=self._apply_folder_to_paths).pack(side="left", padx=(10, 0))

        tk.Checkbutton(f, text="Write caption_tr.txt + caption_en.txt + caption_ua.txt (FULL TEXT)",
                       variable=self.write_files, bg="#2b2b2b", fg="white",
                       selectcolor="#2b2b2b", activebackground="#2b2b2b").pack(anchor="w", padx=16, pady=8)

        grid = tk.Frame(f, bg="#2b2b2b")
        grid.pack(fill="x", padx=16, pady=8)
        for r, (lbl, var) in enumerate([("TR file", self.out_tr), ("EN file", self.out_en), ("UA file", self.out_ua)]):
            tk.Label(grid, text=lbl, bg="#2b2b2b", fg="white").grid(row=r, column=0, sticky="w", padx=6, pady=6)
            tk.Entry(grid, textvariable=var, width=60).grid(row=r, column=1, sticky="ew", padx=6, pady=6)
        grid.columnconfigure(1, weight=1)

        sep = tk.Frame(f, bg="#444444", height=1)
        sep.pack(fill="x", padx=16, pady=12)

        tk.Label(f, text="Session Transcript (everything spoken since Start)", bg="#2b2b2b", fg="white").pack(anchor="w", padx=16, pady=6)
        tk.Checkbutton(f, text="Enable transcript (append)", variable=self.write_transcript,
                       bg="#2b2b2b", fg="white", selectcolor="#2b2b2b",
                       activebackground="#2b2b2b").pack(anchor="w", padx=16, pady=4)
        tk.Checkbutton(f, text="Clear transcript on Start", variable=self.transcript_clear_on_start,
                       bg="#2b2b2b", fg="white", selectcolor="#2b2b2b",
                       activebackground="#2b2b2b").pack(anchor="w", padx=16, pady=4)
        row = tk.Frame(f, bg="#2b2b2b")
        row.pack(fill="x", padx=16, pady=8)
        tk.Label(row, text="Transcript file", bg="#2b2b2b", fg="white").pack(side="left", padx=(0, 8))
        tk.Entry(row, textvariable=self.transcript_path, width=60).pack(side="left", fill="x", expand=True)

        # Controls
        f = self.sections["Controls"]
        tk.Label(f, text="Controls", bg="#2b2b2b", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12)
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(f, textvariable=self.status_var, bg="#2b2b2b", fg="#aaaaaa", font=("Segoe UI", 12)).pack(pady=8)
        tk.Label(f, text=f"Config is saved to: {APPDIR}", bg="#2b2b2b", fg="#aaaaaa").pack(pady=4)

        # Live Output
        f = self.sections["Live Output"]
        tk.Label(f, text="Live Output", bg="#2b2b2b", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12)
        cols = tk.Frame(f, bg="#2b2b2b")
        cols.pack(fill="both", expand=True, padx=16, pady=10)
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)
        cols.columnconfigure(2, weight=1)

        tk.Label(cols, text="Türkçe (STT)", bg="#2b2b2b", fg="white").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Label(cols, text="English (DeepL)", bg="#2b2b2b", fg="white").grid(row=0, column=1, sticky="w", padx=6, pady=6)
        tk.Label(cols, text="Ukrainian (DeepL)", bg="#2b2b2b", fg="white").grid(row=0, column=2, sticky="w", padx=6, pady=6)

        self.tr_text = tk.Text(cols, height=10, wrap="word")
        self.en_text = tk.Text(cols, height=10, wrap="word")
        self.ua_text = tk.Text(cols, height=10, wrap="word")
        self.tr_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.en_text.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        self.ua_text.grid(row=1, column=2, sticky="nsew", padx=6, pady=6)
        cols.rowconfigure(1, weight=1)

        # Log
        f = self.sections["Log"]
        tk.Label(f, text="Log", bg="#2b2b2b", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12)
        self.log = tk.Text(f, height=20, wrap="word")
        self.log.pack(fill="both", expand=True, padx=16, pady=10)

        # Stats
        f = self.sections["Stats"]
        tk.Label(f, text="Stats", bg="#2b2b2b", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12)
        self.stats_spoken_var = tk.StringVar(value="Spoken Minutes: 0.00")
        self.stats_cost_var = tk.StringVar(value="Estimated Daily Cost: $0.00 (OpenAI: $0.00 | DeepL: $0.00)")
        tk.Label(f, textvariable=self.stats_spoken_var, bg="#2b2b2b", fg="#aaaaaa", font=("Segoe UI", 12)).pack(pady=8)
        tk.Label(f, textvariable=self.stats_cost_var, bg="#2b2b2b", fg="#aaaaaa", font=("Segoe UI", 12)).pack(pady=4)


        # --- Section banners (shown inside each tab, under the controls) ---
        # Uses repo files: banner_api.png / banner_caption.png / banner_controls.png / banner_stats.png
        for _sec_name, _sec_frame in self.sections.items():
            if _sec_name in self.section_banners:
                continue
            _lbl = tk.Label(_sec_frame, bg="#2b2b2b")
            _lbl.pack(fill="x", padx=16, pady=(22, 14))
            self.section_banners[_sec_name] = _lbl

    # ---------------- Logging / status ----------------


    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")

    def _set_status(self, s: str):
        if hasattr(self, "status_var"):
            self.status_var.set(s)

    @staticmethod
    def _set_text(widget: tk.Text, text: str):
        widget.delete("1.0", "end")
        widget.insert("1.0", text)

    # ---------------- Devices ----------------
    def _refresh_devices(self):
        self.device_map.clear()
        items = []
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d.get("max_input_channels", 0) > 0:
                    name = d.get("name", f"Device {i}")
                    display = f"{i}: {name}"
                    self.device_map[display] = i
                    items.append(display)
        except Exception as e:
            messagebox.showerror("Device error", str(e))
            return

        self.device_combo["values"] = items
        if items and not self.selected_device.get():
            self.selected_device.set(items[0])
        self._log("Devices refreshed.")

    # ---------------- Config gather/apply ----------------
    def _gather_config(self) -> dict:
        dev_display = self.selected_device.get()
        dev_index = self.device_map.get(dev_display, None)
        return {
            "openai_key": self.openai_key_entry.get(),
            "deepl_key": self.deepl_key_entry.get(),
            "device_index": dev_index,
            "openai_model": self.openai_model.get(),
            "deepl_base": self.deepl_base.get(),
            "sample_rate": int(self.sample_rate.get()),
            "energy_threshold": float(self.energy_threshold.get()),
            "silence_finalize": float(self.silence_finalize.get()),
            "min_utt": float(self.min_utt.get()),
            "max_utt": float(self.max_utt.get()),
            "min_seconds_between_calls": float(self.min_seconds_between_calls.get()),
            "openai_timeout": int(self.openai_timeout.get()),
            "deepl_timeout": int(self.deepl_timeout.get()),
            "write_files": bool(self.write_files.get()),
            "out_tr": self.out_tr.get(),
            "out_en": self.out_en.get(),
            "out_ua": self.out_ua.get(),
            "write_transcript": bool(self.write_transcript.get()),
            "transcript_path": self.transcript_path.get(),
            "transcript_clear_on_start": bool(self.transcript_clear_on_start.get()),
            "special_enabled": bool(self.special_enabled.get()),
            "special_threshold": float(self.special_threshold.get()),
            "special_text": self.special_text.get("1.0", "end"),
            "output_dir": self.output_dir.get(),
        }

    def _apply_config(self, cfg: dict) -> None:
        if isinstance(cfg.get("openai_key"), str):
            self.openai_key_entry.delete(0, "end")
            self.openai_key_entry.insert(0, cfg["openai_key"])
        if isinstance(cfg.get("deepl_key"), str):
            self.deepl_key_entry.delete(0, "end")
            self.deepl_key_entry.insert(0, cfg["deepl_key"])

        dev_index = cfg.get("device_index", None)
        if dev_index is not None:
            for disp, idx in self.device_map.items():
                if idx == dev_index:
                    self.selected_device.set(disp)
                    break

        # output folder
        if isinstance(cfg.get("output_dir"), str) and cfg.get("output_dir"):
            self.output_dir.set(cfg["output_dir"])
        def s(var, key):
            if key in cfg and cfg[key] is not None:
                try:
                    var.set(cfg[key])
                except Exception:
                    pass

        s(self.openai_model, "openai_model")
        s(self.deepl_base, "deepl_base")
        s(self.sample_rate, "sample_rate")
        s(self.energy_threshold, "energy_threshold")
        s(self.silence_finalize, "silence_finalize")
        s(self.min_utt, "min_utt")
        s(self.max_utt, "max_utt")
        s(self.min_seconds_between_calls, "min_seconds_between_calls")
        s(self.openai_timeout, "openai_timeout")
        s(self.deepl_timeout, "deepl_timeout")
        s(self.write_files, "write_files")
        s(self.out_tr, "out_tr")
        s(self.out_en, "out_en")
        s(self.out_ua, "out_ua")
        s(self.write_transcript, "write_transcript")
        s(self.transcript_path, "transcript_path")
        s(self.transcript_clear_on_start, "transcript_clear_on_start")
        s(self.special_enabled, "special_enabled")
        s(self.special_threshold, "special_threshold")

        if isinstance(cfg.get("special_text"), str):
            self.special_text.delete("1.0", "end")
            self.special_text.insert("1.0", cfg["special_text"])

    def _save_config_now(self, show_popup: bool = True):
        try:
            save_secure_config(self._gather_config())
            if show_popup:
                messagebox.showinfo("Saved", f"Encrypted config saved.\nLocation: {CFG_FILE}")
            self._log("Config saved.")
        except PermissionError as e:
            messagebox.showerror("Permission denied", f"{e}\n\nTip: Config is saved under APP FOLDER:\n{APPDIR}")
        except Exception as e:
            messagebox.showerror("Config error", str(e))

    def _load_config_into_ui(self):
        cfg = load_secure_config()
        if not cfg:
            return
        self._apply_config(cfg)
        self._log("Config loaded.")

    # ---------------- Start/Stop + animations ----------------
    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        openai_key = self.openai_key_entry.get().strip()
        deepl_key = self.deepl_key_entry.get().strip()
        if not openai_key:
            messagebox.showwarning("Missing key", "OpenAI API key is required.")
            return
        if not deepl_key:
            messagebox.showwarning("Missing key", "DeepL API key is required.")
            return

        dev_display = self.selected_device.get()
        if dev_display not in self.device_map:
            messagebox.showwarning("Device", "Select a valid input device.")
            return
        device_index = self.device_map[dev_display]

        # Save config on Start (silent)
        try:
            save_secure_config(self._gather_config())
        except Exception:
            pass

        # Prepare outputs (clear)
        try:
            if self.write_files.get():
                atomic_write(self.out_tr.get().strip() or "caption_tr.txt", "", self.output_dir.get())
                atomic_write(self.out_en.get().strip() or "caption_en.txt", "", self.output_dir.get())
                atomic_write(self.out_ua.get().strip() or "caption_ua.txt", "", self.output_dir.get())

            if self.write_transcript.get() and self.transcript_clear_on_start.get():
                atomic_write(self.transcript_path.get().strip() or "session_transcript.txt", "", self.output_dir.get())
        except PermissionError as e:
            messagebox.showerror("Permission denied", f"{e}\n\nTip: Use absolute paths to a writable folder.\nDefault writable folder:\n{APPDIR}")
            return
        except Exception as e:
            messagebox.showerror("File error", str(e))
            return

        self.spoken_seconds = 0.0
        self.deepl_chars = 0

        # NOTE: do not reset output_dir here; keep user selection
        if not (self.output_dir.get() or "").strip():
            self.output_dir.set(BASE_DIR)
        mapping, canon_norm, canon_original = parse_special_names(self.special_text.get("1.0", "end"))

        self.stop_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._set_status("Running…")
        self.running_anim = True
        self._green_glow()
        self._log(f"Starting with device: {dev_display}")

        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(device_index, openai_key, deepl_key, mapping, canon_norm, canon_original),
            daemon=True
        )
        self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        self._set_status("Stopping…")
        self._log("Stop requested.")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.running_anim = False
        self._red_pulse()

    def _green_glow(self):
        if not self.running_anim:
            self.start_btn.configure(bg="#1f6f3f")
            return
        current = self.start_btn.cget("bg")
        new = "#2faa4f" if current == "#1f6f3f" else "#1f6f3f"
        self.start_btn.configure(bg=new)
        self.after(600, self._green_glow)

    def _red_pulse(self):
        for i in range(6):
            self.after(i*150, lambda c="#c0392b": self.stop_btn.configure(bg=c))
            self.after(i*150+75, lambda c="#8b1e1e": self.stop_btn.configure(bg=c))

    # ---------------- Worker loop ----------------
    def _poll_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "tr":
                    self._set_text(self.tr_text, payload)
                elif kind == "en":
                    self._set_text(self.en_text, payload)
                elif kind == "ua":
                    self._set_text(self.ua_text, payload)
                elif kind == "status":
                    self._set_status(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    def _update_stats_labels(self):
        minutes = self.spoken_seconds / 60.0
        openai_cost = minutes * OPENAI_PRICE_PER_MIN
        deepl_cost = (self.deepl_chars / 1_000_000.0) * DEEPL_PRICE_PER_MILLION_CHAR
        total = openai_cost + deepl_cost
        self.stats_spoken_var.set(f"Spoken Minutes: {minutes:.2f}")
        self.stats_cost_var.set(f"Estimated Daily Cost: ${total:.2f} (OpenAI: ${openai_cost:.2f} | DeepL: ${deepl_cost:.2f})")
        self.after(500, self._update_stats_labels)

    def _worker_loop(self, device_index: int, openai_key: str, deepl_key: str, sn_mapping, sn_canon_norm, sn_canon_original):
        q_audio = queue.Queue()
        sr = int(self.sample_rate.get())
        model = self.openai_model.get().strip() or "gpt-4o-transcribe"
        deepl_base = self.deepl_base.get().strip() or "https://api-free.deepl.com"

        seg = AudioSegmenter(
            energy_threshold=float(self.energy_threshold.get()),
            silence_to_finalize=float(self.silence_finalize.get()),
            min_utt_s=float(self.min_utt.get()),
            max_utt_s=float(self.max_utt.get()),
        )

        min_gap = float(self.min_seconds_between_calls.get())
        oa_timeout = int(self.openai_timeout.get())
        dl_timeout = int(self.deepl_timeout.get())

        out_tr_path = (self.out_tr.get().strip() or "caption_tr.txt")
        out_en_path = (self.out_en.get().strip() or "caption_en.txt")
        out_ua_path = (self.out_ua.get().strip() or "caption_ua.txt")

        transcript_path = (self.transcript_path.get().strip() or "session_transcript.txt")
        write_transcript = bool(self.write_transcript.get())

        use_special = bool(self.special_enabled.get())
        threshold = float(self.special_threshold.get())

        last_call_time = 0.0

        def callback(indata, frames, t, status):
            mono = indata[:, 0].copy()
            q_audio.put((time.time(), mono))

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                device=device_index,
                blocksize=int(sr * 0.05),
                callback=callback,
            ):
                self.ui_queue.put(("status", "Running… (listening)"))

                while not self.stop_event.is_set():
                    try:
                        now, chunk = q_audio.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    finalize, audio_seg = seg.feed(chunk, now)
                    if not finalize:
                        continue
                    if audio_seg is None or audio_seg.size < int(sr * 0.4):
                        continue

                    self.spoken_seconds += float(audio_seg.size) / float(sr)

                    gap = now - last_call_time
                    if gap < min_gap:
                        time.sleep(max(0.0, min_gap - gap))

                    self.ui_queue.put(("status", "Transcribing (TR)…"))
                    wav_bytes = to_wav_bytes(audio_seg, sr)

                    try:
                        tr_text = openai_transcribe_tr(wav_bytes, openai_key, model, timeout_s=oa_timeout)
                    except Exception as e:
                        self.ui_queue.put(("log", f"OpenAI error: {e}"))
                        self.ui_queue.put(("status", "Running… (listening)"))
                        last_call_time = time.time()
                        continue

                    last_call_time = time.time()
                    if not tr_text:
                        self.ui_queue.put(("status", "Running… (listening)"))
                        continue

                    if use_special:
                        tr_text = apply_special_names(tr_text, sn_mapping, sn_canon_norm, sn_canon_original, threshold)

                    self.ui_queue.put(("tr", tr_text))
                    if self.write_files.get():
                        try:
                            atomic_write(out_tr_path, tr_text, self.output_dir.get())
                        except Exception as e:
                            self.ui_queue.put(("log", f"TR file write error: {e}"))

                    self.ui_queue.put(("status", "Translating (EN+UA)…"))
                    try:
                        en_text = deepl_translate(tr_text, deepl_key, deepl_base, target="EN", timeout_s=dl_timeout).strip()
                        ua_text = deepl_translate(tr_text, deepl_key, deepl_base, target="UK", timeout_s=dl_timeout).strip()
                    except Exception as e:
                        self.ui_queue.put(("log", f"DeepL error: {e}"))
                        self.ui_queue.put(("status", "Running… (listening)"))
                        continue

                    self.deepl_chars += len(tr_text)

                    if use_special:
                        if en_text:
                            en_text = apply_special_names(en_text, sn_mapping, sn_canon_norm, sn_canon_original, threshold)
                        if ua_text:
                            ua_text = apply_special_names(ua_text, sn_mapping, sn_canon_norm, sn_canon_original, threshold)

                    self.ui_queue.put(("en", en_text))
                    self.ui_queue.put(("ua", ua_text))

                    if self.write_files.get():
                        try:
                            atomic_write(out_en_path, en_text, self.output_dir.get())
                        except Exception as e:
                            self.ui_queue.put(("log", f"EN file write error: {e}"))
                        try:
                            atomic_write(out_ua_path, ua_text, self.output_dir.get())
                        except Exception as e:
                            self.ui_queue.put(("log", f"UA file write error: {e}"))

                    if write_transcript:
                        try:
                            ts = time.strftime("%Y-%m-%d %H:%M:%S")
                            append_write(transcript_path, f"[{ts}] TR: {tr_text}", self.output_dir.get())
                            append_write(transcript_path, f"[{ts}] EN: {en_text}", self.output_dir.get())
                            append_write(transcript_path, f"[{ts}] UA: {ua_text}", self.output_dir.get())
                            append_write(transcript_path, "", self.output_dir.get())
                        except Exception as e:
                            self.ui_queue.put(("log", f"Transcript write error: {e}"))

                    self.ui_queue.put(("status", "Running… (listening)"))

        except Exception as e:
            self.ui_queue.put(("log", f"ERROR: {e}"))
            self.ui_queue.put(("status", "Idle (error)"))
        finally:
            self.ui_queue.put(("status", "Idle"))
            self.ui_queue.put(("log", "Stopped."))

    # ---------------- Close ----------------
    def _on_close(self):
        try:
            if self.worker_thread and self.worker_thread.is_alive():
                self.stop_event.set()
        except Exception:
            pass
        try:
            save_secure_config(self._gather_config())
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    LiveCaptionApp().mainloop()