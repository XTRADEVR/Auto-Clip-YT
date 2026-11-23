"""AutoClip AI - Streamlit.

Link YouTube (atau upload video) -> transkrip -> Gemini memilih momen terbaik
-> crop 9:16 mengikuti wajah -> subtitle karaoke -> klip siap upload.

File utama di Streamlit Cloud: app.py
"""
import base64
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import requests
import streamlit as st

st.set_page_config(page_title="AutoClip AI", page_icon="🎬", layout="wide")

try:
    import yt_dlp
except ImportError:
    st.error("Paket yt-dlp belum terpasang. Pastikan requirements.txt berisi baris: yt-dlp[default]")
    st.stop()


# ------------------------------------------------- komponen opsional (runtime)
# requirements.txt sengaja minimal supaya instalasi tidak gagal. Komponen berat/opsional
# (JS runtime untuk YouTube, OpenCV untuk face tracking) dipasang di sini, dan bila gagal
# aplikasi tetap jalan tanpa fitur tersebut.
def get_cv2():
    try:
        import cv2

        return cv2
    except Exception:  # noqa: BLE001 - ImportError atau konflik versi numpy
        return None


def face_detection_ready() -> bool:
    """True bila OpenCV ada DAN file cascade wajah tersedia (ada di OpenCV 4.x, tidak di 5.x)."""
    cv2 = get_cv2()
    if cv2 is None:
        return False
    try:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        return os.path.exists(path) and not cv2.CascadeClassifier(path).empty()
    except Exception:  # noqa: BLE001
        return False


def pip_install(package: str) -> bool:
    """Pasang paket tanpa dependensi (supaya numpy/streamlit yang sedang jalan tidak terganggu)."""
    cmds = [[sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", package]]
    uv = shutil.which("uv")
    if uv:
        cmds.append([uv, "pip", "install", "--quiet", "--no-deps", "--python", sys.executable, package])
    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
            if proc.returncode == 0:
                importlib.invalidate_caches()
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def find_deno() -> Optional[str]:
    found = shutil.which("deno")
    if found:
        return found
    try:
        import deno  # paket PyPI "deno" (biner Deno)

        path = deno.find_deno_bin()
        return path if path and os.path.exists(path) else None
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner="Menyiapkan komponen tambahan (JS runtime & face tracking)...")
def setup_extras() -> dict:
    deno_path = find_deno()
    if not deno_path and pip_install("deno"):
        deno_path = find_deno()
    if deno_path:  # yt-dlp otomatis memakai Deno bila ada di PATH
        os.environ["PATH"] = os.path.dirname(deno_path) + os.pathsep + os.environ.get("PATH", "")

    face = face_detection_ready()
    if not face and get_cv2() is None and pip_install("opencv-python-headless<5"):
        face = face_detection_ready()
    return {"deno": deno_path, "face": face}


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # noqa: BLE001 - secrets.toml belum dibuat
        return default


# ------------------------------------------------------------------ download
def download(url: str, workdir: Path, cookies: Optional[Path], on_progress: Callable[[float], None]):
    """Unduh video (maks 1080p) + subtitle otomatis YouTube (json3) bila tersedia."""

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(min(d.get("downloaded_bytes", 0) / total, 1.0))

    base = {
        "outtmpl": str(workdir / "source.%(ext)s"),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 30,
        "progress_hooks": [hook],
    }
    if cookies:
        base["cookiefile"] = str(cookies)

    with_subs = {
        **base,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["id", "en"],
        "subtitlesformat": "json3",
    }
    try:
        with yt_dlp.YoutubeDL(with_subs) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception:  # noqa: BLE001
        # Biasanya gagal karena subtitle (mis. HTTP 429) -> ulangi tanpa subtitle.
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=True)

    videos = [p for p in workdir.glob("source.*") if p.suffix != ".json3"]
    if not videos:
        raise RuntimeError("File video tidak ditemukan setelah diunduh.")
    mp4 = [p for p in videos if p.suffix == ".mp4"]
    video = mp4[0] if mp4 else videos[0]
    subs = sorted(workdir.glob("source.*.json3"))
    return video, (subs[0] if subs else None), info


def explain_error(msg: str) -> str:
    low = msg.lower()
    if "not a bot" in low or "sign in to confirm" in low or "confirm you" in low:
        return (
            "YouTube menolak unduhan dari server ini (IP server cloud sering diblokir). "
            "Solusi: unggah cookies.txt dari browser yang login YouTube di sidebar, atau pakai tab "
            "'Upload video'."
        )
    if "429" in low or "too many requests" in low:
        return "YouTube membatasi permintaan (HTTP 429). Tunggu beberapa menit lalu coba lagi, atau pakai cookies.txt."
    if "requested format is not available" in low or "javascript" in low or "challenge" in low:
        return (
            "YouTube butuh JS runtime untuk membaca video ini dan komponen Deno belum aktif. "
            "Klik 'Siapkan ulang komponen' di sidebar, atau pakai tab 'Upload video'."
        )
    if "private video" in low or "video unavailable" in low or "members-only" in low:
        return "Video ini privat, dihapus, atau khusus member sehingga tidak bisa diunduh."
    return "Gagal mengunduh video."


# ----------------------------------------------------------------- transkrip
# Struktur segmen: {"start", "end", "text", "words": [{"w", "start", "end"}]}
def even_words(text: str, start: float, end: float) -> List[dict]:
    toks = text.split()
    if not toks:
        return []
    if end <= start:
        end = start + max(1.5, len(toks) * 0.35)
    step = (end - start) / len(toks)
    return [
        {"w": t, "start": start + i * step, "end": start + (i + 1) * step} for i, t in enumerate(toks)
    ]


def parse_json3(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for e in data.get("events", []):
        segs = e.get("segs")
        if not segs:
            continue
        ev_start = e.get("tStartMs", 0) / 1000
        ev_end = ev_start + e.get("dDurationMs", 2000) / 1000
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if not text:
            continue

        if any("tOffsetMs" in s for s in segs):  # subtitle otomatis: ada waktu per kata
            words = []
            for s in segs:
                w = s.get("utf8", "").strip()
                if w:
                    words.append({"w": w, "start": ev_start + s.get("tOffsetMs", 0) / 1000})
            for i, w in enumerate(words):
                nxt = words[i + 1]["start"] if i + 1 < len(words) else ev_end
                w["end"] = max(min(nxt, w["start"] + 1.2), w["start"] + 0.05)
        else:  # subtitle manual: bagi rata di sepanjang durasi
            words = even_words(text, ev_start, ev_end)
        out.append({"start": ev_start, "end": ev_end, "text": text, "words": words})
    return out


def parse_ts(value) -> float:
    """Terima 12.5, '12.5', '01:30', atau '1:02:03' -> detik."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        secs = 0.0
        for part in str(value).strip().split(":"):
            secs = secs * 60 + float(part)
        return secs
    except ValueError:
        return 0.0


def segments_from_items(items) -> List[dict]:
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        try:
            text = str(it["text"]).strip()
            start, end = parse_ts(it["start"]), parse_ts(it["end"])
        except (KeyError, TypeError):
            continue
        if not text:
            continue
        if end <= start:
            end = start + max(1.5, len(text.split()) * 0.35)
        out.append({"start": start, "end": end, "text": text, "words": even_words(text, start, end)})
    return sorted(out, key=lambda s: s["start"])


def transcribe_gemini(video: Path, workdir: Path, api_key: str, model: str) -> List[dict]:
    """Transkrip lewat Gemini (audio dikirim langsung). Timestamp per segmen, kata dibagi rata."""
    audio = workdir / "audio.mp3"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video.resolve()), "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "32k", "-t", "3000", str(audio)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not audio.exists():
        raise RuntimeError("Gagal mengekstrak audio (video mungkin tanpa suara).")
    if audio.stat().st_size > 14_000_000:
        raise RuntimeError("Audio terlalu besar untuk dikirim ke Gemini.")

    prompt = (
        "Transkripsikan ucapan pada audio ini dalam bahasa aslinya. "
        'Balas HANYA JSON array: [{"start": "MM:SS", "end": "MM:SS", "text": "..."}]. '
        "Tiap segmen 2-6 detik, waktu dihitung dari awal audio. Jangan menulis apa pun di luar JSON."
    )
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/mp3", "data": base64.b64encode(audio.read_bytes()).decode()}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
        },
        timeout=300,
    )
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return segments_from_items(json.loads(text))


def transcribe_whisper(video: Path) -> List[dict]:
    """Fallback lokal. Butuh paket faster-whisper (opsional, tidak ada di requirements)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    model = WhisperModel(os.getenv("WHISPER_MODEL", "small"), compute_type="int8")
    segments, _ = model.transcribe(str(video), vad_filter=True, word_timestamps=True)
    out = []
    for s in segments:
        words = [
            {"w": w.word.strip(), "start": w.start, "end": w.end}
            for w in (s.words or [])
            if w.word.strip()
        ]
        out.append({"start": s.start, "end": s.end, "text": s.text.strip(), "words": words})
    return out


def get_transcript(video: Path, sub: Optional[Path], workdir: Path, api_key: str, model: str) -> List[dict]:
    if sub:
        try:
            transcript = parse_json3(sub)
            if transcript:
                return transcript
        except Exception:  # noqa: BLE001
            pass
    if api_key:
        try:
            transcript = transcribe_gemini(video, workdir, api_key, model)
            if transcript:
                return transcript
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Transkrip lewat Gemini gagal ({str(exc)[:150]}).")
    return transcribe_whisper(video)


# ----------------------------------------------------------------- highlight
def mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def fallback_highlights(duration: float, n: int, min_s: int, max_s: int) -> List[dict]:
    length = float(min(max_s, max(min_s, 45)))
    if duration <= length:
        return [{"start": 0.0, "end": duration, "title": "Klip 1", "reason": "Video pendek"}]
    step = duration / (n + 1)
    result = []
    for i in range(1, n + 1):
        start = max(0.0, min(duration - length, step * i - length / 2))
        result.append(
            {"start": start, "end": start + length, "title": f"Klip {i}", "reason": "Dibagi rata (tanpa AI)"}
        )
    return result


def clean_highlights(items, duration: float, n: int, min_s: int, max_s: int) -> List[dict]:
    cleaned: List[dict] = []
    if not isinstance(items, list):
        return cleaned
    for it in items:
        try:
            start, end = parse_ts(it["start"]), parse_ts(it["end"])
        except (KeyError, TypeError):
            continue
        start = max(0.0, start)
        end = min(duration, end)
        if end - start > max_s:
            end = start + max_s
        if end - start < min_s:
            continue
        if any(start < c["end"] and end > c["start"] for c in cleaned):
            continue
        cleaned.append(
            {
                "start": start,
                "end": end,
                "title": str(it.get("title", f"Klip {len(cleaned) + 1}"))[:80],
                "reason": str(it.get("reason", ""))[:200],
            }
        )
        if len(cleaned) >= n:
            break
    return sorted(cleaned, key=lambda c: c["start"])


def find_highlights(transcript, duration, n, min_s, max_s, api_key, model) -> List[dict]:
    if not api_key or not transcript:
        return fallback_highlights(duration, n, min_s, max_s)

    lines = "\n".join(f"[{mmss(t['start'])}] {t['text']}" for t in transcript)[:90000]
    prompt = (
        "Kamu editor video short-form (TikTok/Reels/Shorts).\n"
        f"Dari transkrip berikut (durasi video {int(duration)} detik), pilih {n} momen paling menarik: "
        "hook kuat, lucu, emosional, insight berharga, atau klimaks cerita.\n"
        f"Aturan: tiap klip {min_s}-{max_s} detik, mulai di awal kalimat dan selesai di akhir kalimat, "
        "tidak tumpang tindih.\n"
        'Balas HANYA JSON array: [{"start": detik, "end": detik, '
        '"title": "judul singkat menarik", "reason": "alasan singkat"}]\n\n'
        f"TRANSKRIP:\n{lines}"
    )
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        items = clean_highlights(json.loads(text), duration, n, min_s, max_s)
        if items:
            return items
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Gemini gagal memilih momen ({str(exc)[:120]}). Memakai pembagian otomatis.")
    return fallback_highlights(duration, n, min_s, max_s)


# ------------------------------------------------------------ subtitle: SRT
def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(transcript, start: float, end: float, path: Path) -> bool:
    blocks = []
    for seg in transcript:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        a = max(seg["start"], start) - start
        b = min(seg["end"], end) - start
        blocks.append(f"{len(blocks) + 1}\n{srt_time(a)} --> {srt_time(b)}\n{seg['text']}\n")
    if blocks:
        path.write_text("\n".join(blocks), encoding="utf-8")
    return bool(blocks)


# --------------------------------------------------------- subtitle: karaoke
def ass_time(t: float) -> str:
    cs = int(round(max(t, 0.0) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(word: str) -> str:
    return word.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def collect_words(transcript, start: float, end: float) -> List[dict]:
    """Ambil kata-kata dalam rentang klip, waktu relatif terhadap awal klip."""
    words = []
    for seg in transcript:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        for w in seg.get("words", []):
            if w["start"] >= start - 0.05 and w["start"] < end:
                words.append(
                    {"w": w["w"], "start": max(w["start"] - start, 0.0), "end": min(w["end"], end) - start}
                )
    words.sort(key=lambda x: x["start"])
    return words


def group_words(words: List[dict], max_words: int = 4, max_chars: int = 24, max_gap: float = 0.7):
    """Kelompokkan kata jadi baris pendek ala Shorts."""
    lines, cur = [], []
    for w in words:
        if cur:
            too_long = len(" ".join(x["w"] for x in cur + [w])) > max_chars
            gap = w["start"] - cur[-1]["end"]
            if len(cur) >= max_words or too_long or gap > max_gap:
                lines.append(cur)
                cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def write_karaoke_ass(transcript, start: float, end: float, path: Path, vertical: bool) -> bool:
    """Tulis file ASS: kata berubah warna (putih -> kuning) tepat saat diucapkan."""
    lines = group_words(collect_words(transcript, start, end))
    if not lines:
        return False

    if vertical:
        res_x, res_y, size, margin_v, outline = 1080, 1920, 76, 420, 6
    else:
        res_x, res_y, size, margin_v, outline = 1920, 1080, 62, 90, 4

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {res_x}\n"
        f"PlayResY: {res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # PrimaryColour = warna setelah diucapkan (kuning), SecondaryColour = sebelum (putih)
        f"Style: Default,DejaVu Sans,{size},&H0000E5FF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},2,2,60,60,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for line in lines:
        l_start = line[0]["start"]
        l_end = max(line[-1]["end"], l_start + 0.2) + 0.15
        parts = []
        for i, w in enumerate(line):
            nxt = line[i + 1]["start"] if i + 1 < len(line) else line[-1]["end"]
            k = max(int(round((nxt - w["start"]) * 100)), 1)
            parts.append(f"{{\\k{k}}}{ass_escape(w['w']).upper()}")
        events.append(f"Dialogue: 0,{ass_time(l_start)},{ass_time(l_end)},Default,,0,0,0,,{' '.join(parts)}")

    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return True


# ------------------------------------------------------------- face tracking
def probe_size(video: Path) -> Tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video.resolve()),
        ],
        capture_output=True, text=True,
    )
    try:
        w, h = proc.stdout.strip().split("\n")[0].split(",")[:2]
        return int(w), int(h)
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Resolusi video tidak terbaca (ffprobe).") from exc


def probe_duration(video: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video.resolve()),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip().split("\n")[0])
    except (ValueError, IndexError):
        return 0.0


def face_knots(video: Path, start: float, end: float, crop_w: int, src_w: int,
               interval: float = 0.75) -> Optional[List[Tuple[float, int]]]:
    """Deteksi wajah tiap `interval` detik -> daftar titik (waktu, posisi x crop).

    Posisi dihaluskan, dan titik baru hanya dibuat bila wajah bergeser cukup jauh,
    sehingga kamera tidak bergetar.
    """
    cv2 = get_cv2()
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        cap.release()
        return None

    scale = min(1.0, 640.0 / src_w)
    times, centers = [], []
    t = 0.0
    duration = end - start
    while t <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000)
        ok, frame = cap.read()
        cx = None
        if ok:
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                x, _, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx = (x + w / 2) / scale
        times.append(t)
        centers.append(cx)
        t += interval
    cap.release()

    known = [c for c in centers if c is not None]
    if not known:
        return None

    last = known[0]  # isi posisi kosong dengan posisi terakhir yang diketahui
    filled = []
    for c in centers:
        last = c if c is not None else last
        filled.append(last)

    smooth, cur = [], filled[0]  # exponential smoothing
    for c in filled:
        cur = cur + 0.4 * (c - cur)
        smooth.append(cur)

    max_x = src_w - crop_w
    xs = [int(min(max(c - crop_w / 2, 0), max_x)) for c in smooth]

    knots = [(times[0], xs[0])]
    for tt, xx in zip(times[1:], xs[1:]):
        if abs(xx - knots[-1][1]) > 0.04 * crop_w:
            knots.append((round(tt, 2), xx))
    return knots


def build_x_expr(knots: List[Tuple[float, int]]) -> str:
    """Ubah titik-titik jadi ekspresi FFmpeg: interpolasi linear antar titik, lalu tahan."""
    if len(knots) == 1:
        return str(int(knots[0][1]))
    terms = []
    for (t0, x0), (t1, x1) in zip(knots, knots[1:]):
        terms.append(f"gte(t,{t0:.2f})*lt(t,{t1:.2f})*({x0}+({x1 - x0})*(t-{t0:.2f})/{t1 - t0:.2f})")
    t_last, x_last = knots[-1]
    terms.append(f"gte(t,{t_last:.2f})*{x_last}")
    return "+".join(terms)


# --------------------------------------------------------------------- potong
def cut_clip(video: Path, transcript, h: dict, outdir: Path, idx: int, vertical: bool,
             sub_mode: str, face_track: bool) -> Path:
    out_name = f"clip_{idx}.mp4"
    filters = []

    if vertical:
        src_w, src_h = probe_size(video)
        crop_w = int(src_h * 9 / 16) // 2 * 2
        if crop_w >= src_w:  # sumber sudah vertikal/kotak: letterbox, jangan crop
            filters.append(
                "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
            )
        else:
            x_expr = str((src_w - crop_w) // 2)
            if face_track:
                try:
                    knots = face_knots(video, h["start"], h["end"], crop_w, src_w)
                    if knots:
                        x_expr = build_x_expr(knots)
                except Exception:  # noqa: BLE001 - gagal deteksi -> crop tengah
                    pass
            filters.append(f"crop={crop_w}:ih:x='{x_expr}':y=0")
            filters.append("scale=1080:1920")

    if sub_mode == "Karaoke (per kata)":
        ass_name = f"clip_{idx}.ass"
        if write_karaoke_ass(transcript, h["start"], h["end"], outdir / ass_name, vertical):
            filters.append(f"ass={ass_name}")
        else:  # tidak ada data per kata -> subtitle standar
            sub_mode = "Standar"
    if sub_mode == "Standar":
        srt_name = f"clip_{idx}.srt"
        if write_srt(transcript, h["start"], h["end"], outdir / srt_name):
            style = "FontSize=14,Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV=70"
            filters.append(f"subtitles={srt_name}:force_style='{style}'")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{h['start']:.2f}",
        "-t", f"{h['end'] - h['start']:.2f}",
        "-i", str(video.resolve()),
    ]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        out_name,
    ]
    # cwd=outdir supaya nama file subtitle tidak perlu di-escape di filter FFmpeg.
    proc = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg gagal: {proc.stderr[-300:]}")
    return outdir / out_name


def fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


# ------------------------------------------------------------------------ UI
extras = setup_extras()

st.title("🎬 AutoClip AI")
st.caption("Ubah video panjang jadi klip pendek siap upload: AI memilih momen terbaik, crop 9:16, subtitle karaoke.")

with st.sidebar:
    st.header("Pengaturan")

    default_key = get_secret("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    model_name = st.text_input("Model Gemini", value="gemini-2.5-flash")
    clip_count = st.slider("Jumlah klip", 1, 10, 3)
    min_s, max_s = st.slider("Durasi klip (detik)", 10, 180, (20, 60))
    vertical = st.checkbox("Format vertikal 9:16", value=True)
    face_track = st.checkbox(
        "Fokus ke wajah (face tracking)",
        value=extras["face"],
        disabled=not (extras["face"] and vertical),
        help="Crop 9:16 mengikuti wajah pembicara. Hanya untuk format vertikal.",
    )
    sub_mode = st.selectbox("Subtitle", ["Karaoke (per kata)", "Standar", "Tanpa subtitle"])
    cookies_file = st.file_uploader(
        "cookies.txt YouTube (opsional)",
        type=["txt"],
        help="Isi bila YouTube menolak unduhan ('Sign in to confirm you're not a bot'). "
             "Bisa juga disimpan permanen di Secrets sebagai YOUTUBE_COOKIES.",
    )

    st.divider()
    st.caption("Status komponen")
    st.caption(("✅" if extras["deno"] else "⚠️") + " JS runtime YouTube (Deno)")
    st.caption(("✅" if extras["face"] else "⚠️") + " Face tracking (OpenCV)")
    st.caption(("✅" if api_key else "⚠️") + " Gemini API key")
    if st.button("Siapkan ulang komponen"):
        setup_extras.clear()
        st.rerun()


def prepare_cookies(workdir: Path) -> Optional[Path]:
    data = None
    if cookies_file is not None:
        data = cookies_file.getvalue()
    else:
        secret = get_secret("YOUTUBE_COOKIES", "")
        if secret:
            data = secret.encode("utf-8")
    if not data:
        return None
    path = workdir / "cookies.txt"
    path.write_bytes(data)
    return path


def run_job(mode: str, url: str, upload) -> None:
    st.session_state.pop("result", None)
    workdir = Path(tempfile.mkdtemp(prefix="autoclip_"))
    bar = st.progress(0.0, text="Memulai...")

    def update(frac: float, msg: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        if mode == "link":
            update(0.02, "Mengunduh video...")
            video, sub, info = download(
                url, workdir, prepare_cookies(workdir), lambda p: update(0.02 + p * 0.28, "Mengunduh video...")
            )
            title = info.get("title", "")
            duration = float(info.get("duration") or 0)
        else:
            update(0.02, "Menyimpan video...")
            video = workdir / ("source" + (Path(upload.name).suffix.lower() or ".mp4"))
            video.write_bytes(upload.getvalue())
            sub, title = None, Path(upload.name).stem
            duration = probe_duration(video)
        if duration <= 0:
            raise RuntimeError("Durasi video tidak terbaca (mungkin siaran langsung atau file rusak).")

        update(0.32, "Membuat transkrip...")
        transcript = get_transcript(video, sub, workdir, api_key, model_name)
        if not transcript:
            st.warning("Transkrip tidak tersedia: klip dibagi rata dan tanpa subtitle. "
                       "Isi Gemini API key agar transkrip dan pemilihan momen otomatis berjalan.")

        update(0.45, "AI memilih momen terbaik...")
        highlights = find_highlights(transcript, duration, clip_count, min_s, max_s, api_key, model_name)

        clips = []
        for i, h in enumerate(highlights, 1):
            update(0.5 + 0.48 * (i - 1) / len(highlights), f"Memotong klip {i}/{len(highlights)}...")
            path = cut_clip(video, transcript, h, workdir, i, vertical, sub_mode, face_track and vertical)
            clips.append({**h, "data": path.read_bytes()})

        update(1.0, "Selesai")
        st.session_state["result"] = {"title": title, "vertical": vertical, "clips": clips}
    except yt_dlp.utils.DownloadError as exc:
        st.error(explain_error(str(exc)))
        with st.expander("Detail error"):
            st.code(str(exc)[:1500])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Terjadi kesalahan: {str(exc)[:400]}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def ffmpeg_ok() -> bool:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True
    st.error("FFmpeg tidak ditemukan. Di Streamlit Cloud, tambahkan file packages.txt berisi satu baris: ffmpeg")
    return False


tab_link, tab_file = st.tabs(["🔗 Link YouTube", "📁 Upload video"])

with tab_link:
    url = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Buat Klip dari Link", type="primary", key="run_link"):
        u = url.strip()
        if not u.lower().startswith(("http://", "https://")) or not any(
            d in u.lower() for d in ("youtube.com/", "youtu.be/")
        ):
            st.error("Masukkan link YouTube yang valid.")
        elif min_s >= max_s:
            st.error("Durasi minimum harus lebih kecil dari maksimum.")
        elif ffmpeg_ok():
            run_job("link", u, None)
    st.caption("Jika YouTube menolak unduhan dari server, unggah cookies.txt di sidebar atau pakai tab Upload video.")

with tab_file:
    upload = st.file_uploader("Video (mp4, mov, mkv, webm)", type=["mp4", "mov", "mkv", "webm", "m4v"])
    if st.button("Buat Klip dari File", type="primary", key="run_file"):
        if upload is None:
            st.error("Pilih file video terlebih dahulu.")
        elif min_s >= max_s:
            st.error("Durasi minimum harus lebih kecil dari maksimum.")
        elif ffmpeg_ok():
            run_job("file", "", upload)

result = st.session_state.get("result")
if result:
    st.subheader(result["title"] or "Hasil klip")
    cols_per_row = 3 if result["vertical"] else 2
    clips = result["clips"]
    for row_start in range(0, len(clips), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (i, c) in zip(cols, list(enumerate(clips, 1))[row_start:row_start + cols_per_row]):
            with col:
                st.video(c["data"])
                st.markdown(f"**{c['title']}**")
                st.caption(f"{fmt_time(c['start'])} - {fmt_time(c['end'])} · {c['reason']}")
                st.download_button(
                    "Unduh klip",
                    data=c["data"],
                    file_name=f"clip_{i}.mp4",
                    mime="video/mp4",
                    key=f"dl_{i}",
                )

st.divider()
st.caption("Gunakan hanya untuk video milik sendiri atau yang kamu punya izin untuk dipotong dan dipublikasikan ulang.")
