"""
Ꮇʏᴛʜɪᴄ ᴀɪ — single file, powered by Google's Gemini API or a local Ollama model.

Usage (Gemini — default, needs a free API key):
    1. pip install flask requests
    2. Set your API key:
         Mac/Linux:   export GEMINI_API_KEY="your-key-here"
         Windows:     set GEMINI_API_KEY=your-key-here
    3. python ai_chat.py
    4. Open http://localhost:5000 in your browser

Get a FREE API key (no credit card needed) at https://aistudio.google.com/apikey

Usage (Ollama — fully local, no API key or internet needed):
    1. Install Ollama from https://ollama.com and make sure it's running
       (`ollama serve`, or it may already be running as a background service)
    2. Pull a model, e.g.:  ollama pull llama3.1
    3. Set the provider:
         Mac/Linux:   export AI_PROVIDER=ollama
         Windows:     set AI_PROVIDER=ollama
       Optional overrides:
         OLLAMA_URL   (default: http://localhost:11434)
         OLLAMA_MODEL (default: llama3.1)
    4. python ai_chat.py
    5. Open http://localhost:5000 in your browser

Supabase (optional — for accounts/conversation storage across restarts & devices):
    Set these as environment variables (never hardcode secrets in this file):
         SUPABASE_URL   e.g. https://xxxxx.supabase.co
         SUPABASE_KEY   your Supabase *secret* key (server-side only, keeps full DB access)
    If unset, the app falls back to storing conversations as local JSON files in chat_data/.

Features:
- Login/register (real accounts, hashed passwords, stored in chat_data/users.json)
- Multi-conversation chat with sidebar, saved per-account, survives restarts
- File/image upload (attach an image or text file to a message)
- Web search grounding (Gemini can search Google for current info — Gemini only)
- Streaming responses (text appears word-by-word)
- Switchable AI backend: Google Gemini (cloud) or Ollama (local, private, free)
- No rate limiting — unlimited messages
"""

import os
import json
import uuid
import time
import base64
import requests
from flask import (
    Flask, request, jsonify, Response, session,
    stream_with_context
)

PROVIDER = os.environ.get("AI_PROVIDER", "auto").strip().lower()
# "auto"        = round-robin: Groq → OpenRouter → HuggingFace (all work on servers)
# "gemini"      = Google Gemini only (free tier only works locally, not on Render)
# "groq"        = Groq only
# "openrouter"  = OpenRouter only
# "huggingface" = Hugging Face only
# "ollama"      = local Ollama only

# --- API Keys (hardcoded fallbacks — override via environment variables) ------
# WARNING: don't commit a file with real keys to a public GitHub repo.
# Set these as environment variables on Render instead.
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",    "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",      "")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY",  "")
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY","")
HF_API_KEY        = os.environ.get("HF_API_KEY",        "")
# NanoBanana API (nanobananaapi.ai) — powers "Ghibli Me" image editing so it can
# actually transform the user's uploaded photo (image-to-image), not just
# generate a generic image from text. Get a key at https://nanobananaapi.ai/api-key
# and set it as an environment variable — never hardcode it here.
NANO_BANANA_API_KEY = os.environ.get("NANO_BANANA_API_KEY", "")
NANO_BANANA_BASE     = "https://api.nanobananaapi.ai/api/v1/nanobanana"

# --- Model names -------------------------------------------------------------
GEMINI_MODEL      = "gemini-2.5-flash"
GROQ_MODEL        = os.environ.get("GROQ_MODEL",        "llama-3.1-8b-instant")
OPENROUTER_MODEL  = os.environ.get("OPENROUTER_MODEL",  "google/gemma-3-4b-it:free")
HF_MODEL          = os.environ.get("HF_MODEL",          "mistralai/Mistral-7B-Instruct-v0.3")
CEREBRAS_MODEL    = os.environ.get("CEREBRAS_MODEL",    "gpt-oss-120b")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL",       "llama3.1")
OLLAMA_URL        = os.environ.get("OLLAMA_URL",         "http://localhost:11434").rstrip("/")

GEMINI_STREAM_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"
)

# keep old name for compatibility with existing references below
API_KEY = GEMINI_API_KEY
MODEL   = GEMINI_MODEL
SYSTEM_PROMPT = (
    "You are Ꮇʏᴛʜɪᴄ ᴀɪ, a smart and friendly AI assistant made by Aarav Singh. "
    "If asked who made you, say you are Ꮇʏᴛʜɪᴄ ᴀɪ made by Aarav Singh — say it once naturally, never repeat it unprompted. "
    "Never mention Google, Groq, OpenRouter, HuggingFace, Meta, Mistral, Anthropic, or any AI company as your creator or backend. "
    "You can help with anything: questions, writing, coding, math, ideas, or just chatting. "
    "When writing code, always wrap it in markdown code blocks with the language name. "
    "LANGUAGE: Always reply ENTIRELY in the same language the user's message is written in — "
    "never mix two languages in a single reply. If they write in Hindi, reply fully in Hindi. "
    "If they write in English, reply fully in English (do not slip into Hindi or any other language "
    "partway through, even if source information you know is in a different language — translate it "
    "into the reply language first). If they mix languages themselves, match their mix. "
    "Never force English on the user. "
    "TOOL USE: Never write out fake tool calls, function names, or JSON like {\"query\": ...} in your reply — "
    "those are internal mechanisms the user must never see. If you don't actually have live web access, "
    "just answer from what you know and say your information may not be fully up to date, instead of "
    "pretending to search. "
    "ANTI-REPETITION RULES — follow strictly every reply: "
    "1. NEVER restate or echo back what the user just said. Jump straight to the answer. "
    "2. NEVER start replies with filler like Great question, Sure, Of course, Absolutely, Certainly. "
    "3. NEVER repeat information already given earlier in the conversation. Build on it. "
    "4. Be direct and natural — like a knowledgeable friend, not a customer service bot. "
    "5. Keep answers concise unless the user asks for detail."
)

# Extra instruction appended ONLY for Gemini, which actually has a real google_search tool wired up.
# Other providers (Groq/Cerebras/OpenRouter/HF/Ollama) have no real search access, so telling them
# "you have search" makes them hallucinate fake tool-call JSON into the visible reply — hence this
# is kept separate from the base SYSTEM_PROMPT above.
GEMINI_SEARCH_ADDENDUM = (
    " WEB SEARCH: You have access to Google Search. When the user asks about current events, "
    "live prices, news, sports scores, weather, or anything that needs up-to-date information, "
    "use the search tool to find the answer. Do not say you cannot search the web. When you use "
    "search results, translate/summarize them into the reply language — never paste a mix of languages."
)

app = Flask(__name__)


def _persistent_secret_key():
    """A stable Flask secret key across restarts/workers so each visitor's
    session cookie (which stores their anonymous user_id) doesn't get
    invalidated every time the server restarts. Without this, a random key
    was generated per-process, which silently "lost" every conversation
    tied to the old session on every restart/redeploy/worker respawn —
    the classic cause of "it only ever shows 1 chat".

    Priority: FLASK_SECRET_KEY env var > a key file persisted next to this
    script. Setting FLASK_SECRET_KEY explicitly is STRONGLY recommended for
    any real deployment (Render, etc.), since some hosts wipe local disk
    between deploys, which would defeat the file fallback too.
    """
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "chat_data")
    os.makedirs(data_dir, exist_ok=True)
    key_path = os.path.join(data_dir, "flask_secret.key")
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            existing = f.read().strip()
        if existing:
            return existing
    new_key = str(uuid.uuid4())
    with open(key_path, "w") as f:
        f.write(new_key)
    return new_key


app.secret_key = _persistent_secret_key()

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

# --- Temporary public image hosting (for NanoBanana image-to-image editing) --
# NanoBanana's /generate endpoint needs a publicly reachable image URL for
# edit-mode requests — it can't accept a raw base64 upload directly. So when a
# user uploads a selfie for "Ghibli Me", we stash the bytes here in memory
# under a random id and serve them back at /api/temp-image/<id>, giving
# NanoBanana's servers a URL they can fetch. Entries expire on their own after
# a short TTL so memory doesn't grow unbounded.
_TEMP_IMAGES = {}
_TEMP_IMAGE_TTL_SECONDS = 30 * 60  # 30 minutes


def _store_temp_image(raw_bytes, mime_type):
    # Opportunistic cleanup of anything past its TTL
    cutoff = time.time() - _TEMP_IMAGE_TTL_SECONDS
    for k in [k for k, v in _TEMP_IMAGES.items() if v["created"] < cutoff]:
        _TEMP_IMAGES.pop(k, None)
    img_id = uuid.uuid4().hex
    _TEMP_IMAGES[img_id] = {"data": raw_bytes, "mime_type": mime_type or "image/png", "created": time.time()}
    return img_id


def nano_banana_submit(prompt, image_urls=None, num_images=1):
    """Submits a NanoBanana generation/edit task. Returns (task_id, error)."""
    if not NANO_BANANA_API_KEY:
        return None, "NanoBanana API key not configured"
    payload = {
        "prompt": prompt,
        "type": "IMAGETOIAMGE" if image_urls else "TEXTTOIAMGE",
        "numImages": num_images,
    }
    if image_urls:
        payload["imageUrls"] = image_urls
    try:
        resp = requests.post(
            f"{NANO_BANANA_BASE}/generate",
            headers={"Authorization": f"Bearer {NANO_BANANA_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("code") == 200:
            return data["data"]["taskId"], None
        return None, data.get("msg") or f"NanoBanana error ({resp.status_code})"
    except (requests.RequestException, ValueError) as e:
        return None, str(e)


def nano_banana_poll(task_id, max_wait=180, interval=3):
    """Polls a NanoBanana task until it succeeds/fails/times out.
    Returns (result_image_url, error)."""
    headers = {"Authorization": f"Bearer {NANO_BANANA_API_KEY}"}
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{NANO_BANANA_BASE}/record-info",
                params={"taskId": task_id}, headers=headers, timeout=15,
            )
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            return None, str(e)
        flag = data.get("successFlag")
        if flag == 1:
            result = data.get("response") or {}
            url = result.get("resultImageUrl")
            if not url and isinstance(result.get("resultImageUrls"), list) and result["resultImageUrls"]:
                url = result["resultImageUrls"][0]
            if not url:
                return None, "NanoBanana returned no result image"
            return url, None
        if flag in (2, 3):
            return None, data.get("errorMessage") or "NanoBanana generation failed"
        time.sleep(interval)
    return None, "NanoBanana generation timed out"

# --- Supabase config ---------------------------------------------------------
# Never hardcode real keys here — always set these as environment variables:
#   SUPABASE_URL  -> your project URL, e.g. https://xxxxx.supabase.co
#   SUPABASE_KEY  -> your Supabase *secret* (service-role-style) key
# The secret key grants full database access, so it must only ever live in
# server-side environment variables/secret storage, never in source code,
# client-side JS, or anything committed to a public repo.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def sb(path):
    return f"{SUPABASE_URL}/rest/v1/{path}"


# --- User accounts (Supabase: users table) -----------------------------------

def current_username():
    """Each visitor gets a unique anonymous ID stored in their browser cookie.
    No login required — conversations are private per browser session."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["user_id"]


def login_required(view):
    """No-op decorator kept so all @login_required routes still work unchanged."""
    def wrapped(*args, **kwargs):
        current_username()  # ensure session id is set
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped


# --- Supabase / file storage helpers ----------------------------------------

def list_conversations(username):
    if not SUPABASE_URL:
        return _list_conversations_file(username)
    try:
        r = requests.get(
            sb(f"conversations?username=eq.{username}&order=updated_at.desc&select=id,title,updated_at"),
            headers=sb_headers(), timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def load_conversation(username, conv_id):
    if not SUPABASE_URL:
        return _load_conversation_file(username, conv_id)
    try:
        r = requests.get(
            sb(f"conversations?id=eq.{conv_id}&username=eq.{username}"),
            headers=sb_headers(), timeout=10,
        )
        if r.status_code == 200:
            rows = r.json()
            if rows:
                row = rows[0]
                return {
                    "title": row["title"],
                    "updated_at": row["updated_at"],
                    "messages": row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"]),
                }
    except Exception:
        pass
    return None


def save_conversation(username, conv_id, data):
    data["updated_at"] = time.time()
    if not SUPABASE_URL:
        _save_conversation_file(username, conv_id, data)
        return
    try:
        headers = {**sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
        requests.post(
            sb("conversations"),
            headers=headers,
            json={
                "id": conv_id,
                "username": username,
                "title": data.get("title", "New chat"),
                "updated_at": data["updated_at"],
                "messages": data.get("messages", []),
            },
            timeout=15,
        )
    except Exception:
        pass


def delete_conversation(username, conv_id):
    if not SUPABASE_URL:
        _delete_conversation_file(username, conv_id)
        return
    try:
        requests.delete(
            sb(f"conversations?id=eq.{conv_id}&username=eq.{username}"),
            headers=sb_headers(), timeout=10,
        )
    except Exception:
        pass


# --- Local file fallbacks for when Supabase is not configured ----------------
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_DATA_DIR = _os.path.join(_BASE_DIR, "chat_data")
_os.makedirs(_DATA_DIR, exist_ok=True)

def _user_conv_dir(username):
    path = _os.path.join(_DATA_DIR, "conversations", username)
    _os.makedirs(path, exist_ok=True)
    return path

def _conv_file(username, conv_id):
    return _os.path.join(_user_conv_dir(username), f"{conv_id}.json")

def _list_conversations_file(username):
    folder = _user_conv_dir(username)
    convs = []
    for fname in _os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        path = _os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            convs.append({"id": fname[:-5], "title": d.get("title", "New chat"), "updated_at": d.get("updated_at", 0)})
        except Exception:
            continue
    convs.sort(key=lambda c: c["updated_at"], reverse=True)
    return convs

def _load_conversation_file(username, conv_id):
    path = _conv_file(username, conv_id)
    if not _os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_conversation_file(username, conv_id, data):
    with open(_conv_file(username, conv_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _delete_conversation_file(username, conv_id):
    path = _conv_file(username, conv_id)
    if _os.path.exists(path):
        _os.remove(path)


def make_title(first_message):
    title = (first_message or "Attachment").strip().replace("\n", " ")
    return title[:40] + ("…" if len(title) > 40 else "")


# --- HTML pages ----------------------------------------------------------

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="theme-color" content="#10a37f">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Mythic AI">
<meta name="description" content="Mythic AI - Smart AI assistant by Aarav Singh">
<link rel="manifest" href="/manifest.json">
<link rel="icon" type="image/png" sizes="192x192" href="/icon.png">
<link rel="icon" type="image/png" sizes="512x512" href="/icon-512.png">
<link rel="shortcut icon" href="/favicon.ico">
<link rel="apple-touch-icon" href="/icon.png">
<title>Ꮖỿᴛнᴄ ᴀɪ</title>
<style>

  :root {
    --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a;
    --text:#ececec; --muted:#8e8ea0; --accent:#10a37f;
    --accent-dim:#1a3a30; --user-bubble:#2a2a2a; --user-text:#ececec;
    --ai-bubble:#1a1a1a; --sidebar-w:260px; --msg-font-size:14.5px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif; overflow:hidden; }
  .layout { display:flex; height:100vh; }

  /* Light theme override */
  body.theme-light {
    --bg:#f7f7f8; --panel:#ffffff; --border:#e3e3e6;
    --text:#1f1f1f; --muted:#6b6b76; --accent-dim:#e3f5ef;
    --user-bubble:#eef0f2; --user-text:#1f1f1f; --ai-bubble:#ffffff;
  }

  /* Sidebar */
  #sidebar { width:var(--sidebar-w); flex-shrink:0; background:var(--panel);
    border-right:1px solid var(--border); display:flex; flex-direction:column;
    transition:margin-left .2s ease; }
  #sidebar.hidden { margin-left:calc(-1 * var(--sidebar-w)); }
  #new-chat-btn { margin:12px; padding:10px 14px; background:var(--accent); color:#fff;
    border:none; border-radius:8px; font-size:13.5px; font-weight:600; cursor:pointer; text-align:left; }
  #new-chat-btn:hover { opacity:.9; }
  #conv-list { flex:1; overflow-y:auto; padding:0 8px; display:flex; flex-direction:column; gap:2px; }
  .conv-item { display:flex; align-items:center; justify-content:space-between; gap:6px;
    padding:9px 10px; border-radius:7px; cursor:pointer; font-size:13px; color:var(--muted); }
  .conv-item:hover { background:var(--accent-dim); color:var(--text); }
  .conv-item.active { background:var(--accent-dim); color:var(--accent); font-weight:500; }
  .conv-item .title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
  .conv-item .rename-btn { opacity:0; background:none; border:none; color:var(--muted);
    cursor:pointer; font-size:12px; padding:2px 5px; flex-shrink:0; touch-action:manipulation; }
  .conv-item .del-btn { opacity:0; background:none; border:none; color:var(--muted);
    cursor:pointer; font-size:13px; padding:2px 5px; flex-shrink:0; touch-action:manipulation; }
  .conv-item:hover .rename-btn { opacity:1; }
  .conv-item:hover .del-btn { opacity:1; }
  .conv-item .rename-btn:hover { color:var(--accent); }
  .conv-item .del-btn:hover { color:#ef4444; }
  #sidebar-footer { padding:12px; font-size:11px; color:var(--muted); border-top:1px solid var(--border); }

  /* Main */
  .app { display:flex; flex-direction:column; height:100vh; flex:1; min-width:0; }
  header { padding:calc(14px + env(safe-area-inset-top)) 20px 14px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; gap:10px;
    background:var(--bg); position:relative; z-index:20; }
  header .left { display:flex; align-items:center; gap:10px; min-width:0; }
  header .right { display:flex; align-items:center; gap:8px; flex-shrink:0; }
  header button { touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #sidebar-toggle { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0; }
  #sidebar-toggle:hover { background:var(--panel); }
  header h1 { font-size:16px; font-weight:700; color:var(--accent); margin:0; }
  #name-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #name-btn:hover { background:var(--panel); }
  #settings-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #settings-btn:hover { background:var(--panel); }
  #export-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #export-btn:hover { background:var(--panel); }

  /* Fullscreen bar — sits above the input row so it's always reachable on mobile,
     away from any notch/status-bar area that can swallow top-corner taps. */
  #fullscreen-btn { display:flex; align-items:center; justify-content:center; gap:6px;
    width:100%; max-width:760px; margin:0 auto 8px; padding:9px 12px;
    background:var(--panel); border:1px solid var(--border); border-radius:10px;
    color:var(--muted); font-size:13px; cursor:pointer; touch-action:manipulation;
    -webkit-tap-highlight-color:transparent; }
  #fullscreen-btn:hover { color:var(--text); border-color:var(--accent); }
  #fullscreen-btn.active { color:var(--accent); border-color:var(--accent); }
  #fullscreen-icon { font-size:15px; }

  /* Fallback for browsers without a real Fullscreen API (iOS Safari, some in-app webviews):
     hide the sidebar toggle/header chrome so the chat fills the screen. */
  body.pseudo-fullscreen #sidebar-toggle,
  body.pseudo-fullscreen header .left h1 { display:none; }
  body.pseudo-fullscreen header { padding-top:calc(6px + env(safe-area-inset-top)); padding-bottom:6px; }

  /* Name modal */
  #name-modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55);
    z-index:200; align-items:center; justify-content:center; }
  #name-modal-overlay.show { display:flex; }
  #name-modal { background:var(--bg); border:1px solid var(--border); border-radius:14px;
    padding:22px; width:90%; max-width:360px; box-shadow:0 10px 40px rgba(0,0,0,.3); }
  #name-modal h3 { margin:0 0 6px; font-size:16px; color:var(--text); }
  #name-modal p { margin:0 0 14px; font-size:12.5px; color:var(--muted); }
  #name-input { width:100%; box-sizing:border-box; padding:10px 12px; border-radius:8px;
    border:1.5px solid var(--border); background:var(--panel); color:var(--text);
    font-size:14.5px; outline:none; }
  #name-input:focus { border-color:var(--accent); }
  #name-modal-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:16px; }
  #name-modal-actions button { padding:8px 14px; border-radius:8px; font-size:13px;
    cursor:pointer; border:1px solid var(--border); background:none; color:var(--text); }
  #name-cancel-btn:hover { background:var(--panel); }
  #name-save-btn { background:var(--accent); color:#fff; border-color:var(--accent); }
  #name-save-btn:hover { opacity:.9; }
  #clear-btn { background:none; border:1px solid var(--border); color:var(--muted);
    font-size:12px; padding:6px 12px; border-radius:6px; cursor:pointer; flex-shrink:0; }
  #clear-btn:hover { background:var(--panel); }

  /* Settings modal */
  #settings-modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55);
    z-index:200; align-items:center; justify-content:center; }
  #settings-modal { background:var(--bg); border:1px solid var(--border); border-radius:14px;
    padding:22px; width:92%; max-width:420px; max-height:86vh; overflow-y:auto;
    box-shadow:0 10px 40px rgba(0,0,0,.3); }
  #settings-modal h3 { margin:0 0 4px; font-size:17px; color:var(--text); }
  #settings-modal p.sub { margin:0 0 16px; font-size:12.5px; color:var(--muted); }
  .settings-section { margin-bottom:16px; }
  .settings-section label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; font-weight:600; }
  .settings-row { display:flex; gap:8px; flex-wrap:wrap; }
  .settings-choice { flex:1; min-width:80px; padding:8px 10px; border-radius:8px; border:1.5px solid var(--border);
    background:var(--panel); color:var(--muted); cursor:pointer; font-size:12.5px; font-family:inherit; text-align:center; }
  .settings-choice:hover { border-color:var(--accent); }
  #accent-color-input { width:44px; height:34px; border:1.5px solid var(--border); border-radius:8px;
    background:var(--panel); cursor:pointer; padding:2px; }
  #font-size-slider { width:100%; accent-color:var(--accent); }
  #font-size-label { font-size:12px; color:var(--muted); }
  .settings-select { width:100%; padding:9px 10px; border-radius:8px; border:1.5px solid var(--border);
    background:var(--panel); color:var(--text); font-size:13px; font-family:inherit; outline:none; }
  .settings-select:focus { border-color:var(--accent); }
  #custom-instructions-input { width:100%; box-sizing:border-box; padding:9px 12px; border-radius:8px;
    border:1.5px solid var(--border); background:var(--panel); color:var(--text);
    font-size:13px; font-family:inherit; outline:none; resize:vertical; min-height:60px; }
  #custom-instructions-input:focus { border-color:var(--accent); }
  #settings-close-btn { width:100%; margin-top:6px; background:var(--accent); color:#fff; border:none;
    border-radius:10px; padding:11px; font-size:14px; font-weight:700; cursor:pointer; font-family:inherit; }
  #settings-close-btn:hover { opacity:.9; }

  /* Message bubble density controlled by settings */
  body.bubble-compact .msg { padding:7px 11px; border-radius:12px; }
  body.bubble-compact #messages { gap:8px; }
  body.bubble-comfortable .msg { padding:11px 15px; border-radius:18px; }
  body.bubble-comfortable #messages { gap:16px; }
  body.bubble-spacious .msg { padding:16px 20px; border-radius:22px; }
  body.bubble-spacious #messages { gap:24px; }

  /* Messages */
  #messages-wrap { flex:1; overflow-y:auto; position:relative; }
  #messages { padding:24px 20px; display:flex; flex-direction:column; gap:16px;
    max-width:760px; margin:0 auto; width:100%; min-height:100%; }
  .msg { max-width:80%; padding:11px 15px; border-radius:18px; line-height:1.6;
    font-size:var(--msg-font-size); white-space:pre-wrap; word-wrap:break-word; }
  .msg.user { align-self:flex-end; background:var(--user-bubble); color:var(--user-text);
    border-bottom-right-radius:4px; }
  .msg.ai { align-self:flex-start; background:var(--ai-bubble); color:var(--text);
    border-bottom-left-radius:4px; }
  .msg.error { align-self:center; background:#fef2f2; border:1px solid #fecaca;
    color:#dc2626; font-size:13px; border-radius:10px; }
  .msg img { max-width:100%; border-radius:10px; display:block; margin-top:8px; }
  .attach-chip { font-size:11.5px; opacity:.75; margin-bottom:4px; }

  /* Message row wraps the bubble + its action buttons (copy / regenerate) */
  .msg-row { display:flex; flex-direction:column; max-width:80%; }
  .msg-row.user { align-self:flex-end; align-items:flex-end; }
  .msg-row.ai { align-self:flex-start; align-items:flex-start; }
  .msg-row.error { align-self:center; align-items:center; max-width:90%; }
  .msg-row .msg { max-width:100%; }
  .msg-actions { display:flex; gap:4px; margin-top:3px; opacity:0; transition:opacity .15s;
    height:22px; }
  .msg-row:hover .msg-actions, .msg-row:focus-within .msg-actions { opacity:1; }
  .msg-actions button { background:none; border:none; color:var(--muted); cursor:pointer;
    font-size:12px; padding:2px 7px; border-radius:5px; touch-action:manipulation;
    -webkit-tap-highlight-color:transparent; }
  .msg-actions button:hover { background:var(--panel); color:var(--text); }
  .msg-timestamp { font-size:10.5px; color:var(--muted); margin-top:2px; }
  .empty-state { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    text-align:center; color:var(--muted); }
  .empty-state h2 { font-size:22px; font-weight:700; color:var(--accent); margin-bottom:8px; }
  .empty-state p { font-size:14px; }
  .typing { align-self:flex-start; display:flex; gap:5px; padding:14px 16px;
    background:var(--ai-bubble); border-radius:18px; border-bottom-left-radius:4px; }
  .typing span { width:7px; height:7px; border-radius:50%; background:var(--muted);
    animation:blink 1.2s infinite ease-in-out; }
  .typing span:nth-child(2) { animation-delay:.2s; }
  .typing span:nth-child(3) { animation-delay:.4s; }
  @keyframes blink { 0%,80%,100%{opacity:.2} 40%{opacity:1} }

  /* Scroll to bottom */
  #scroll-btn { position:fixed; bottom:130px; right:24px; width:36px; height:36px;
    border-radius:50%; background:var(--accent); color:#fff; border:none; cursor:pointer;
    font-size:18px; display:none; align-items:center; justify-content:center;
    box-shadow:0 2px 8px rgba(0,0,0,.15); z-index:10; }
  #scroll-btn.show { display:flex; }

  /* Image preview */
  .gen-img { max-width:320px; border-radius:12px; display:block; margin-top:8px; }

  /* Input area */
  #pending-attach { max-width:760px; margin:0 auto; width:100%; padding:6px 20px 0;
    display:none; align-items:center; gap:8px; font-size:12.5px; color:var(--muted); }
  #pending-attach.show { display:flex; }
  #pending-attach button { background:none; border:none; color:var(--muted); cursor:pointer; }
  .input-area { padding:10px 20px 16px; border-top:1px solid var(--border);
    background:var(--bg); max-width:760px; margin:0 auto; width:100%; }
  .input-row { display:flex; gap:8px; align-items:flex-end; background:var(--panel);
    border:1.5px solid var(--border); border-radius:14px; padding:8px 10px; }
  .input-row:focus-within { border-color:var(--accent); }
  .tool-btn { background:none; border:none; color:var(--muted); cursor:pointer;
    width:36px; height:36px; border-radius:8px; font-size:18px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  .tool-btn:hover { background:var(--accent-dim); color:var(--accent); }
  .tool-btn.active { color:var(--accent); }
  textarea { flex:1; resize:none; background:transparent; border:none; color:var(--text);
    font-size:14.5px; font-family:inherit; line-height:1.4; max-height:140px;
    outline:none; padding:4px 0; }
  textarea::placeholder { color:var(--muted); }
  #send-btn { background:var(--accent); color:#fff; border:none; border-radius:10px;
    width:36px; height:36px; font-size:18px; cursor:pointer; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #send-btn:disabled { background:var(--accent-dim); color:var(--muted); cursor:not-allowed; }
  #send-btn.generating { background:#ef4444; }
  #send-btn.generating:hover { opacity:.9; }
  #voice-btn.listening { color:#ef4444; animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* Speaking indicator */
  #speaking-indicator { display:none; align-items:center; gap:6px; font-size:12px;
    color:var(--accent); padding:4px 0; }
  #speaking-indicator.show { display:flex; }
  #stop-speak-btn { background:none; border:1px solid var(--border); color:var(--muted);
    font-size:11px; padding:2px 8px; border-radius:4px; cursor:pointer; }
  .quick-btn { background:var(--panel); border:1px solid var(--border); color:var(--text);
    font-size:12.5px; padding:6px 14px; border-radius:20px; cursor:pointer;
    transition:all .15s ease; white-space:nowrap; font-family:inherit; touch-action:manipulation; }
  .quick-btn:hover { background:var(--accent-dim); border-color:var(--accent); color:var(--accent); }

  #messages-wrap::-webkit-scrollbar, #conv-list::-webkit-scrollbar { width:6px; }
  #messages-wrap::-webkit-scrollbar-thumb, #conv-list::-webkit-scrollbar-thumb
    { background:var(--border); border-radius:4px; }
  #sidebar-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55);
    z-index:99; -webkit-tap-highlight-color:transparent; }

  @media(max-width:768px) {
    :root { --sidebar-w: 78vw; }

    /* Sidebar slides in as overlay — never pushes content */
    #sidebar { position:fixed; top:0; left:0; z-index:100; height:100%;
      height:-webkit-fill-available; width:var(--sidebar-w) !important;
      transform:translateX(0); transition:transform .25s ease;
      box-shadow:4px 0 24px rgba(0,0,0,.5); }
    #sidebar.hidden { transform:translateX(-105%); margin-left:0 !important; }

    /* Show overlay when sidebar open */
    #sidebar-overlay { display:block; }

    /* Main app always takes full width */
    .app { width:100% !important; flex:1; }

    header { padding:calc(10px + env(safe-area-inset-top)) 12px 10px; }
    header h1 { font-size:14px; }
    #sidebar-toggle { width:38px; height:38px; font-size:14px; }
    #name-btn { width:38px; height:38px; font-size:14px; }
    #settings-btn { width:38px; height:38px; font-size:14px; }
    #export-btn { width:38px; height:38px; font-size:14px; }
    #clear-btn { font-size:11px; padding:8px 10px; min-height:38px; }
    #speak-toggle { font-size:11px; padding:5px 8px; }
    #fullscreen-btn { font-size:12.5px; padding:10px 12px; }

    #messages-wrap { overflow-y:auto; -webkit-overflow-scrolling:touch; }
    #messages { padding:14px 10px; gap:12px; max-width:100%; }
    .msg { max-width:90%; font-size:14px; padding:10px 12px; }
    .msg-row { max-width:90%; }
    .msg-actions { opacity:1; height:26px; } /* no hover on touch — keep always visible */
    .msg-actions button { font-size:13px; padding:4px 9px; min-width:30px; min-height:26px; }

    .input-area { padding:8px 10px max(10px,env(safe-area-inset-bottom)); }
    .input-row { padding:6px 8px; }
    textarea { font-size:16px; } /* 16px prevents iOS zoom */
    .tool-btn { width:34px; height:34px; font-size:17px; }
    #send-btn { width:34px; height:34px; font-size:16px; }

    .empty-state h2 { font-size:19px; }
    .empty-state p { font-size:13px; }
    #scroll-btn { bottom:80px; right:12px; width:34px; height:34px; }

    #new-chat-btn { margin:10px; padding:10px 12px; font-size:13.5px; }
    .conv-item { padding:10px 8px; font-size:13px; min-height:44px; }
    .conv-item .rename-btn { opacity:1; }
    .conv-item .del-btn { opacity:1; }
    #sidebar-footer { font-size:11px; padding:10px 12px; }
  }

  @media(max-width:380px) {
    :root { --sidebar-w: 88vw; }
    .msg { font-size:13.5px; }
    header h1 { font-size:13px; }
    #speak-toggle { display:none; }
  }

</style>
</head>
<body>
<div class="layout">
  <div id="sidebar-overlay" style="display:none;position:fixed;inset:0;background:#0007;z-index:99"></div>
  <div id="sidebar">
    <button id="new-chat-btn">+ New chat</button>
    <div id="conv-list"></div>
    <div id="sidebar-footer">Ꮖỿᴛнᴄ ᴀɪ &middot; by Aarav Singh</div>
  </div>
  <div class="app">
    <header>
      <div class="left">
        <button id="sidebar-toggle" title="Toggle sidebar">&#9776;</button>
        <h1>Ꮖỿᴛнᴄ ᴀɪ</h1>
        <span id="vip-badge" style="display:none;background:linear-gradient(135deg,#f5c542,#e0a800);color:#1a1a1a;font-size:10.5px;font-weight:800;padding:3px 8px;border-radius:10px;letter-spacing:.3px;">VIP</span>
        <select id="model-select" title="Select model" style="background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:5px 8px;font-size:12px;cursor:pointer;outline:none;max-width:130px;font-family:inherit;">
          <option value="mythic-1" data-vip="0">Mythic 1</option>
          <option value="mythic-2" data-vip="0" selected>Mythic 2</option>
          <option value="mythic-3" data-vip="0">Mythic 3</option>
          <option value="mythic-vip" data-vip="1">Mythic VIP &#x1F512;</option>
        </select>
      </div>
      <div class="right">
        <button id="install-btn" title="Install Mythic AI" style="display:none;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;white-space:nowrap;touch-action:manipulation;">&#8595; Install</button>
        <button id="fullscreen-btn" type="button" title="Fullscreen" style="display:flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:6px;background:none;border:1px solid var(--border);color:var(--muted);font-size:15px;cursor:pointer;flex-shrink:0;">
          <span id="fullscreen-icon">&#9974;</span>
        </button>
        <button id="name-btn" title="What should Ꮖỿᴛнᴄ ᴀɪ call you?">&#128578;</button>
        <button id="settings-btn" title="Settings">&#9881;</button>
        <button id="export-btn" title="Export this chat">&#8595;</button>
        <button id="clear-btn">Delete chat</button>
      </div>
    </header>

    <div id="messages-wrap">
      <div id="messages">
        <div class="empty-state" id="empty-state">
          <h2>Ꮖỿᴛнᴄ ᴀɪ</h2>
          <p>Ask me anything, generate images, or just chat &#128075;</p>
        </div>
      </div>
    </div>

    <button id="scroll-btn" title="Scroll to bottom">&#8595;</button>

    <div id="pending-attach">
      &#128206; <span id="pending-attach-name"></span>
      <button id="pending-attach-remove">&#10005;</button>
    </div>

    <div id="speaking-indicator">
      &#128266; Speaking...
      <button id="stop-speak-btn">Stop</button>
    </div>

    <!-- Notification permission banner -->
    <div id="notif-banner" style="display:none;align-items:center;justify-content:space-between;gap:10px;background:linear-gradient(135deg,var(--accent-dim),rgba(16,163,127,.15));border:1px solid var(--accent);border-radius:12px;padding:10px 14px;max-width:760px;margin:8px auto 0;width:calc(100% - 40px);flex-wrap:wrap;">
      <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:0;">
        <span style="font-size:20px;flex-shrink:0;">&#128276;</span>
        <div style="min-width:0;">
          <div style="font-size:13px;font-weight:600;color:var(--text);">Get notified when Ꮖỿᴛнᴄ ᴀɪ replies</div>
          <div style="font-size:11.5px;color:var(--muted);margin-top:1px;">Even when you switch to another tab</div>
        </div>
      </div>
      <div style="display:flex;gap:6px;flex-shrink:0;">
        <button id="notif-banner-allow" type="button" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:7px 14px;font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;white-space:nowrap;">Allow &#128276;</button>
        <button id="notif-banner-dismiss" type="button" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:7px 10px;font-size:12.5px;cursor:pointer;font-family:inherit;">&#10005;</button>
      </div>
    </div>

    <!-- Quick action buttons -->
    <div id="quick-actions" style="display:flex;gap:8px;padding:6px 20px 0;max-width:760px;margin:0 auto;width:100%;flex-wrap:wrap;">
      <button class="quick-btn" id="img-gen-btn">&#127912; Image</button>
      <button class="quick-btn" id="ghibli-btn">&#127807; Ghibli Me</button>
      <button class="quick-btn" id="homework-btn">&#128218; Homework</button>
      <button class="quick-btn" id="weather-btn">&#127780; Weather</button>
      <button class="quick-btn" id="search-btn">&#128269; Search</button>
    </div>

    <!-- INPUT AREA: wrapped in .input-area so it always sits at the bottom of the flex column -->
    <div class="input-area">
      <form id="chat-form">
        <div class="input-row">
          <input type="file" id="file-input" accept="image/*,.txt,.md,.csv,.json,.pdf" style="display:none">
          <button class="tool-btn" id="attach-btn" type="button" title="Attach file">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
          </button>
          <input type="file" id="camera-input" accept="image/*" capture="environment" style="display:none">
          <button class="tool-btn" id="camera-btn" type="button" title="Take photo">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
          </button>
          <textarea id="input" rows="1" placeholder="Message Ꮖỿᴛнᴄ ᴀɪ..."></textarea>
          <button class="tool-btn" id="voice-btn" type="button" title="Voice input">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
          </button>
          <button id="send-btn" type="submit" title="Send">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
          </button>
        </div>
      </form>
    </div>
  </div>
</div>

<div id="name-modal-overlay">
  <div id="name-modal">
    <h3>What should Ꮖỿᴛнᴄ ᴀɪ call you?</h3>
    <p>Enter your preferred name — Ꮖỿᴛнᴄ ᴀɪ will use it when it talks to you.</p>
    <input type="text" id="name-input" maxlength="60" placeholder="e.g. Aarav" autocomplete="off">
    <div id="name-modal-actions">
      <button id="name-cancel-btn" type="button">Cancel</button>
      <button id="name-save-btn" type="button">Save</button>
    </div>
  </div>
</div>

<!-- Settings Modal -->
<div id="settings-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:200;align-items:center;justify-content:center;">
  <div id="settings-modal" style="background:var(--bg);border:1px solid var(--border);border-radius:14px;padding:22px;width:92%;max-width:480px;max-height:86vh;overflow-y:auto;box-shadow:0 10px 40px rgba(0,0,0,.3);">
    <h3 style="margin:0 0 4px;font-size:17px;">Settings</h3>
    <p style="margin:0 0 16px;font-size:12.5px;color:var(--muted);">Customize Ꮖỿᴛнᴄ ᴀɪ. Saved on this device.</p>

    <div class="settings-section"><label>Theme</label><div class="settings-row">
      <button class="settings-choice" data-group="theme" data-value="dark">&#127769; Dark</button>
      <button class="settings-choice" data-group="theme" data-value="light">&#9728; Light</button>
      <button class="settings-choice" data-group="theme" data-value="system">&#128187; System</button>
    </div></div>

    <div class="settings-section"><label>Accent color</label>
      <input type="color" id="accent-color-input" value="#10a37f" style="width:44px;height:34px;border:1.5px solid var(--border);border-radius:8px;background:var(--panel);cursor:pointer;padding:2px;">
    </div>

    <div class="settings-section"><label>Font size — <span id="font-size-label">14.5px</span></label>
      <input type="range" id="font-size-slider" min="12" max="20" step="0.5" value="14.5" style="width:100%;accent-color:var(--accent);">
    </div>

    <div class="settings-section"><label>Bubble spacing</label><div class="settings-row">
      <button class="settings-choice" data-group="bubble" data-value="compact">Compact</button>
      <button class="settings-choice" data-group="bubble" data-value="comfortable">Comfortable</button>
      <button class="settings-choice" data-group="bubble" data-value="spacious">Spacious</button>
    </div></div>

    <div class="settings-section"><label>Reply tone</label>
      <select id="tone-select" style="width:100%;padding:9px 10px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);font-size:13px;font-family:inherit;outline:none;">
        <option value="default">Default</option><option value="formal">Formal</option>
        <option value="casual">Casual</option><option value="funny">Funny</option>
        <option value="professional">Professional</option>
      </select>
    </div>

    <div class="settings-section"><label>Reply length</label>
      <select id="length-select" style="width:100%;padding:9px 10px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);font-size:13px;font-family:inherit;outline:none;">
        <option value="default">Default</option><option value="short">Short</option>
        <option value="medium">Medium</option><option value="long">Long</option>
      </select>
    </div>

    <div class="settings-section"><label>Custom instructions</label>
      <textarea id="custom-instructions-input" rows="2" placeholder="e.g. Always answer in bullet points" style="width:100%;padding:9px 12px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);font-size:13px;font-family:inherit;outline:none;resize:vertical;min-height:52px;"></textarea>
    </div>

    <div class="settings-section" style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px;">
      <label>&#128266; Read-aloud language</label>
      <select id="voice-language-select" style="width:100%;padding:9px 10px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);font-size:13px;font-family:inherit;outline:none;margin-bottom:10px;"></select>
      <label>&#127897; Read-aloud voice</label>
      <select id="voice-select" style="width:100%;padding:9px 10px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);font-size:13px;font-family:inherit;outline:none;"></select>
      <div id="voice-hint" style="font-size:11px;color:var(--muted);margin-top:6px;"></div>
    </div>

    <div class="settings-section" style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px;">
      <label style="display:flex;align-items:center;justify-content:space-between;">
        <span>&#128276; Reply notifications</span>
        <button id="notif-toggle-btn" type="button" style="background:none;border:1.5px solid var(--border);color:var(--muted);border-radius:20px;padding:5px 14px;font-size:12px;cursor:pointer;font-family:inherit;">Enable</button>
      </label>
      <div id="notif-status" style="font-size:11.5px;color:var(--muted);margin-top:6px;"></div>
    </div>

    <!-- API Key Manager -->
    <div class="settings-section" id="api-key-section" style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px;">
      <label>&#128273; My API Keys</label>
      <p style="font-size:11.5px;color:var(--muted);margin:4px 0 10px;">Add your own Groq or Cerebras keys. Select one as active to use it instead of the server default.</p>
      <div id="api-key-list" style="display:flex;flex-direction:column;gap:6px;margin-bottom:10px;"></div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <select id="api-key-provider-select" style="flex:0 0 auto;padding:7px 10px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);font-size:12px;font-family:inherit;outline:none;">
          <option value="groq">Groq</option><option value="cerebras">Cerebras</option>
        </select>
        <input id="api-key-label-input" type="text" maxlength="30" placeholder="Label (e.g. My Key)" style="flex:1;min-width:80px;padding:7px 10px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);font-size:12px;font-family:inherit;outline:none;">
        <input id="api-key-value-input" type="password" maxlength="200" placeholder="Paste API key..." style="flex:2;min-width:100px;padding:7px 10px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);font-size:12px;font-family:inherit;outline:none;">
        <button id="api-key-add-btn" type="button" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:7px 14px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;white-space:nowrap;">+ Add</button>
      </div>
      <div id="api-key-error" style="font-size:11.5px;color:#ef4444;margin-top:6px;display:none;"></div>
    </div>

    <button id="settings-close-btn" type="button" style="width:100%;margin-top:6px;background:var(--accent);color:#fff;border:none;border-radius:10px;padding:11px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">Done</button>
  </div>
</div>

<!-- Ghibli Selfie Modal -->
<div id="ghibli-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:440px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">&#127807; Ghibli Me</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Upload your photo and get a Studio Ghibli-style version</p>
    <div id="ghibli-upload-area" style="border:2px dashed var(--border);border-radius:12px;padding:24px;text-align:center;cursor:pointer;margin-bottom:12px;transition:border-color .2s;">
      <div style="font-size:36px;margin-bottom:8px;">&#128248;</div>
      <div style="font-size:13px;color:var(--muted);">Click to upload your photo<br><span style="font-size:11px;">or drag &amp; drop</span></div>
      <input type="file" id="ghibli-file-input" accept="image/*" style="display:none">
    </div>
    <div id="ghibli-preview-wrap" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="ghibli-preview" style="max-width:100%;max-height:180px;border-radius:10px;border:2px solid var(--accent);">
      <div style="font-size:11px;color:var(--muted);margin-top:4px;">Your photo &#10003;</div>
    </div>
    <div style="margin-bottom:12px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Ghibli Style:</label>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Spirited Away style, soft watercolor anime art" style="padding:8px;border-radius:8px;border:1.5px solid var(--accent);background:var(--accent-dim);color:var(--accent);cursor:pointer;font-size:12px;font-family:inherit;">&#127754; Spirited Away</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, My Neighbor Totoro style, soft forest anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">&#127795; Totoro Forest</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Howl's Moving Castle style, fantasy anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">&#127984; Howl's Castle</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Princess Mononoke style, nature anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">&#128058; Mononoke</button>
      </div>
    </div>
    <input id="ghibli-extra" type="text" placeholder="Add details (optional): e.g. forest background, sunset..." style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;margin-bottom:12px;font-family:inherit;">
    <div id="ghibli-result-wrap" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="ghibli-result" style="max-width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);">
      <button id="ghibli-download-btn" style="margin-top:8px;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px 18px;font-size:13px;cursor:pointer;font-family:inherit;">&#8595; Download</button>
    </div>
    <div id="ghibli-loading" style="display:none;text-align:center;padding:20px;">
      <div style="font-size:32px;margin-bottom:8px;">&#127912;</div>
      <div style="color:var(--muted);font-size:13px;">Creating your Ghibli portrait...<br><span style="font-size:11px;">This takes 15–60 seconds</span></div>
    </div>
    <div id="ghibli-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>
    <div style="display:flex;gap:8px;">
      <button id="ghibli-generate-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">&#10024; Create Ghibli Art</button>
      <button id="ghibli-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">&#10005;</button>
    </div>
  </div>
</div>

<!-- Image Generation Modal -->
<div id="img-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:440px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">&#127912; Generate Image</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Describe what you want to see</p>
    <textarea id="img-prompt" rows="3" placeholder="e.g. a cozy cabin in a snowy forest, golden hour lighting" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:13px;outline:none;margin-bottom:12px;font-family:inherit;resize:vertical;"></textarea>
    <div style="margin-bottom:14px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Style (optional):</label>
      <select id="img-style" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;font-family:inherit;">
        <option value="">&#10024; Auto (recommended)</option>
        <option value="photorealistic, hyperrealistic DSLR photography, 8K resolution, cinematic">&#128247; Photorealistic</option>
        <option value="professional book cover design, award-winning layout, elegant typography">&#128218; Book Cover</option>
        <option value="Studio Ghibli anime style, soft watercolor, vibrant colors, beautiful">&#127807; Anime / Ghibli</option>
        <option value="digital painting, fantasy concept art, epic lighting, deviantart">&#127917; Fantasy Art</option>
        <option value="watercolor painting, soft pastel, dreamy, artistic brushstrokes">&#128396; Watercolor</option>
        <option value="3D render, Octane render, ultra realistic, physically based rendering">&#129522; 3D Render</option>
        <option value="flat vector illustration, minimalist, clean lines, modern design">&#128208; Minimalist</option>
        <option value="oil painting, impressionist, rich textures, museum quality">&#128444; Oil Painting</option>
        <option value="cinematic film still, dramatic lighting, movie poster quality, 35mm">&#127916; Cinematic</option>
        <option value="pixel art, retro 8-bit style, vibrant palette, game art">&#128377; Pixel Art</option>
        <option value="pencil sketch, detailed graphite drawing, fine art, black and white">&#9999; Pencil Sketch</option>
        <option value="logo design, professional brand identity, clean, scalable vector">&#127991; Logo / Brand</option>
      </select>
    </div>
    <div id="img-loading" style="display:none;text-align:center;padding:20px;"><div style="font-size:32px;margin-bottom:8px;">&#127912;</div><div style="color:var(--muted);font-size:13px;">Generating...<br><span style="font-size:11px;opacity:.7;">Auto-enhancing prompt</span></div></div>
    <div id="img-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>
    <div id="img-result" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="img-output" style="max-width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);cursor:zoom-in;">
      <div style="display:flex;gap:8px;margin-top:8px;">
        <button id="img-download-btn" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px;font-size:13px;cursor:pointer;font-family:inherit;">&#8595; Download</button>
        <button id="img-copy-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px;font-size:13px;cursor:pointer;font-family:inherit;">&#128203; Copy</button>
        <button id="img-fullscreen-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px;font-size:13px;cursor:pointer;font-family:inherit;">&#9974; View</button>
      </div>
    </div>
    <div style="display:flex;gap:8px;">
      <button id="img-generate-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">&#10024; Generate</button>
      <button id="img-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">&#10005;</button>
    </div>
  </div>
</div>

<div id="img-viewer-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:400;align-items:center;justify-content:center;cursor:zoom-out;">
  <img id="img-viewer-img" style="max-width:94%;max-height:94%;border-radius:8px;">
</div>

<!-- Weather Modal -->
<div id="weather-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:460px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">&#127780; Weather</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Search any city, or use your current location</p>
    <div style="display:flex;gap:8px;margin-bottom:12px;">
      <input id="weather-city" type="text" placeholder="Search city or place..." autocomplete="off" style="flex:1;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:13px;outline:none;font-family:inherit;">
      <button id="weather-search-btn" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:0 14px;font-size:14px;cursor:pointer;">&#128269;</button>
      <button id="weather-location-btn" title="Use my location" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:0 12px;font-size:14px;cursor:pointer;">&#128205;</button>
    </div>
    <div id="weather-loading" style="display:none;text-align:center;padding:20px;"><div style="font-size:32px;margin-bottom:8px;">&#127758;</div><div style="color:var(--muted);font-size:13px;">Fetching weather...</div></div>
    <div id="weather-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>
    <div id="weather-result" style="display:none;"><div id="weather-content"></div></div>
    <div style="display:flex;justify-content:flex-end;margin-top:14px;">
      <button id="weather-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:10px 16px;font-size:14px;cursor:pointer;">Close</button>
    </div>
  </div>
</div>

<script>

const messagesWrap = document.getElementById('messages-wrap');
const messagesEl   = document.getElementById('messages');
const form         = document.getElementById('chat-form');
const input        = document.getElementById('input');
const sendBtn      = document.getElementById('send-btn');
const clearBtn     = document.getElementById('clear-btn');
const convListEl   = document.getElementById('conv-list');
const newChatBtn   = document.getElementById('new-chat-btn');
const sidebarToggle= document.getElementById('sidebar-toggle');
const fullscreenBtn= document.getElementById('fullscreen-btn');
const nameBtn       = document.getElementById('name-btn');
const modelSelect   = document.getElementById('model-select');

let selectedModel = 'mythic-2';
let vipUnlocked   = false;

function showVipModal() {
  const existing = document.getElementById('vip-modal-overlay');
  if (existing) { existing.style.display = 'flex'; return; }
  const overlay = document.createElement('div');
  overlay.id = 'vip-modal-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;';
  overlay.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:90%;max-width:340px;">
    <div style="font-size:22px;margin-bottom:6px;">🔒 VIP Access</div>
    <div style="color:var(--muted);font-size:13px;margin-bottom:16px;">Mythic VIP is for VIP users only.</div>
    <input id="vip-pw-in" type="password" placeholder="VIP password" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:14px;outline:none;margin-bottom:8px;font-family:inherit;">
    <div id="vip-pw-err" style="color:#ef4444;font-size:12px;display:none;margin-bottom:8px;">Wrong password.</div>
    <div style="display:flex;gap:8px;">
      <button id="vip-pw-ok" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;font-size:14px;font-weight:600;cursor:pointer;">Unlock</button>
      <button id="vip-pw-cancel" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">Cancel</button>
    </div></div>`;
  document.body.appendChild(overlay);
  const pwIn = overlay.querySelector('#vip-pw-in'), pwErr = overlay.querySelector('#vip-pw-err');
  pwIn.focus();
  overlay.querySelector('#vip-pw-cancel').addEventListener('click', () => {
    overlay.style.display = 'none';
    modelSelect.value = selectedModel;
  });
  overlay.querySelector('#vip-pw-ok').addEventListener('click', async () => {
    try {
      const r = await fetch('/api/vip-unlock', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: pwIn.value.trim() })
      });
      const d = await r.json();
      if (d.success) {
        vipUnlocked = true;
        overlay.style.display = 'none';
        selectedModel = 'mythic-vip';
        modelSelect.value = 'mythic-vip';
        const opt = modelSelect.querySelector('option[value="mythic-vip"]');
        if (opt) opt.textContent = 'Mythic VIP ✨';
      } else {
        pwErr.style.display = 'block'; pwIn.value = ''; pwIn.focus();
      }
    } catch {
      pwErr.textContent = 'Network error — try again.';
      pwErr.style.display = 'block';
    }
  });
  pwIn.addEventListener('keydown', e => { if (e.key === 'Enter') overlay.querySelector('#vip-pw-ok').click(); });
}

// Populate model list from the backend (falls back to the static HTML options if it fails)
(async () => {
  try {
    const [mr, vr] = await Promise.all([
      fetch('/api/models').then(r => r.json()),
      fetch('/api/vip-status').then(r => r.json()),
    ]);
    vipUnlocked = !!vr.vip;
    if (mr && Array.isArray(mr.models) && mr.models.length) {
      modelSelect.innerHTML = '';
      mr.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.vip ? (vipUnlocked ? m.name.replace('🔒', '✨') : m.name) : m.name;
        opt.dataset.vip = m.vip ? '1' : '0';
        if (m.id === mr.default) opt.selected = true;
        modelSelect.appendChild(opt);
      });
      selectedModel = mr.default || selectedModel;
    }
  } catch {
    // Backend didn't respond — the static <option> list already in the HTML still works fine.
  }
})();

modelSelect.addEventListener('change', () => {
  const opt = modelSelect.options[modelSelect.selectedIndex];
  if (opt && opt.dataset.vip === '1' && !vipUnlocked) {
    showVipModal();
  } else {
    selectedModel = modelSelect.value;
  }
});

const nameModalOverlay = document.getElementById('name-modal-overlay');
const nameInput     = document.getElementById('name-input');
const nameCancelBtn = document.getElementById('name-cancel-btn');
const nameSaveBtn   = document.getElementById('name-save-btn');
const exportBtn     = document.getElementById('export-btn');
const sidebar      = document.getElementById('sidebar');
const fileInput    = document.getElementById('file-input');
const attachBtn    = document.getElementById('attach-btn');
const cameraInput  = document.getElementById('camera-input');
const cameraBtn    = document.getElementById('camera-btn');
const voiceBtn     = document.getElementById('voice-btn');
const pendingAttach= document.getElementById('pending-attach');
const pendingName  = document.getElementById('pending-attach-name');
const pendingRemove= document.getElementById('pending-attach-remove');
const scrollBtn    = document.getElementById('scroll-btn');
const speakingIndicator = document.getElementById('speaking-indicator');
const stopSpeakBtn = document.getElementById('stop-speak-btn');

let activeConvId = null;
let pendingFile  = null;
let recognition  = null;
let currentUtterance = null;

// --- Scroll button ---
messagesWrap.addEventListener('scroll', () => {
  const nearBottom = messagesWrap.scrollHeight - messagesWrap.scrollTop - messagesWrap.clientHeight < 120;
  scrollBtn.classList.toggle('show', !nearBottom);
});
scrollBtn.addEventListener('click', () => {
  messagesWrap.scrollTo({ top: messagesWrap.scrollHeight, behavior: 'smooth' });
});
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesWrap.scrollTo({ top: messagesWrap.scrollHeight, behavior: 'smooth' });
  });
}

function clearEmptyState() {
  const es = document.getElementById('empty-state');
  if (es) es.remove();
}
function showEmptyState() {
  messagesEl.innerHTML = '<div class="empty-state" id="empty-state"><h2>Ꮇʏᴛʜɪᴄ ᴀɪ</h2><p>Ask me anything, generate images, or just chat 👋</p></div>';
}

function addMessage(role, text, attachment) {
  clearEmptyState();
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;

  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (attachment) {
    const chip = document.createElement('div');
    chip.className = 'attach-chip';
    chip.textContent = '📎 ' + attachment.name;
    div.appendChild(chip);
    if (attachment.mimeType && attachment.mimeType.startsWith('image/') && attachment.dataBase64) {
      const img = document.createElement('img');
      img.src = 'data:' + attachment.mimeType + ';base64,' + attachment.dataBase64;
      div.appendChild(img);
    }
  }
  const textNode = document.createElement('div');
  textNode.className = 'msg-text';
  textNode.textContent = text;
  div.appendChild(textNode);
  row.appendChild(div);

  if (role === 'user' || role === 'ai') {
    row.appendChild(buildMsgActions(row, textNode, role));
  }

  messagesEl.appendChild(row);
  scrollToBottom();
  return textNode;
}

function buildMsgActions(row, textNode, role) {
  const actions = document.createElement('div');
  actions.className = 'msg-actions';

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.className = 'copy-btn';
  copyBtn.title = 'Copy';
  copyBtn.textContent = '📋';
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(textNode.textContent);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = textNode.textContent;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); } catch {}
      ta.remove();
    }
    const orig = copyBtn.textContent;
    copyBtn.textContent = '✓';
    setTimeout(() => { copyBtn.textContent = orig; }, 1200);
  });
  actions.appendChild(copyBtn);

  if (role === 'ai') {
    const regenBtn = document.createElement('button');
    regenBtn.type = 'button';
    regenBtn.className = 'regen-btn';
    regenBtn.title = 'Regenerate response';
    regenBtn.textContent = '↻';
    regenBtn.addEventListener('click', () => regenerateLast(row));
    actions.appendChild(regenBtn);
  }
  return actions;
}

function addImageMessage(role, base64, caption) {
  clearEmptyState();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (caption) {
    const cap = document.createElement('div');
    cap.textContent = caption;
    cap.style.marginBottom = '8px';
    div.appendChild(cap);
  }
  const img = document.createElement('img');
  img.className = 'gen-img';
  img.src = 'data:image/png;base64,' + base64;
  div.appendChild(img);
  messagesEl.appendChild(div);
  scrollToBottom();
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'typing'; div.id = 'typing-indicator';
  div.innerHTML = '<span></span><span></span><span></span>';
  messagesEl.appendChild(div);
  scrollToBottom();
}
function hideTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

// --- Text-to-speech ---
function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const plain = text.replace(/[#*`_~>]/g, '').trim();
  if (!plain) return;
  currentUtterance = new SpeechSynthesisUtterance(plain);
  currentUtterance.rate = 1.05;
  currentUtterance.onstart = () => speakingIndicator.classList.add('show');
  currentUtterance.onend = () => speakingIndicator.classList.remove('show');
  currentUtterance.onerror = () => speakingIndicator.classList.remove('show');
  window.speechSynthesis.speak(currentUtterance);
}
stopSpeakBtn.addEventListener('click', () => {
  window.speechSynthesis && window.speechSynthesis.cancel();
  speakingIndicator.classList.remove('show');
});

// --- Voice input ---
function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { voiceBtn.title = 'Voice not supported in this browser'; return; }
  recognition = new SR();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  let finalTranscript = '';
  recognition.onstart  = () => { voiceBtn.classList.add('active', 'listening'); finalTranscript = ''; };
  recognition.onresult = (e) => {
    finalTranscript = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) finalTranscript += e.results[i][0].transcript;
      else input.value = e.results[i][0].transcript;
    }
    if (finalTranscript) input.value = finalTranscript;
  };
  recognition.onend = () => {
    voiceBtn.classList.remove('active', 'listening');
    if (input.value.trim()) form.requestSubmit();
  };
  recognition.onerror = () => voiceBtn.classList.remove('active', 'listening');
}
setupVoice();
voiceBtn.addEventListener('click', () => {
  if (!recognition) { alert('Voice input is not supported in this browser. Try Chrome.'); return; }
  if (voiceBtn.classList.contains('listening')) { recognition.stop(); return; }
  recognition.start();
});

// --- File / camera attach ---
function handleFileSelect(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    const base64  = dataUrl.split(',')[1];
    pendingFile = { name: file.name, mimeType: file.type || 'application/octet-stream', dataBase64: base64 };
    pendingName.textContent = file.name;
    pendingAttach.classList.add('show');
  };
  reader.readAsDataURL(file);
}
attachBtn.addEventListener('click', () => fileInput.click());
cameraBtn.addEventListener('click', () => cameraInput.click());
fileInput.addEventListener('change', () => handleFileSelect(fileInput.files[0]));
cameraInput.addEventListener('change', () => handleFileSelect(cameraInput.files[0]));
pendingRemove.addEventListener('click', () => {
  pendingFile = null;
  fileInput.value = '';
  cameraInput.value = '';
  pendingAttach.classList.remove('show');
});

// --- Image generation detection ---
const IMAGE_KEYWORDS = /\b(generate|create|draw|make|paint|render|show me|ghibli|anime|realistic|cartoon|portrait|landscape|art|artwork|image of|picture of|photo of|illustration)\b/i;
async function tryGenerateImage(prompt) {
  try {
    const r = await fetch('/api/generate-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const d = await r.json();
    if (d.image) {
      addImageMessage('ai', d.image, '');
      return true;
    }
  } catch {}
  return false;
}

// --- Conversations ---
async function loadConversationList() {
  try {
    const r = await fetch('/api/conversations');
    const d = await r.json();
    const convs = d.conversations || [];
    convListEl.innerHTML = '';
    convs.forEach(c => {
      const item = document.createElement('div');
      item.className = 'conv-item' + (c.id === activeConvId ? ' active' : '');
      item.innerHTML = '<span class="title"></span>'
        + '<button class="rename-btn" title="Rename">✎</button>'
        + '<button class="del-btn" title="Delete">✕</button>';
      item.querySelector('.title').textContent = c.title;
      item.addEventListener('click', (e) => {
        if (!e.target.classList.contains('del-btn') && !e.target.classList.contains('rename-btn')) openConversation(c.id);
      });
      item.querySelector('.rename-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        const newTitle = prompt('Rename chat:', c.title);
        if (!newTitle || !newTitle.trim() || newTitle.trim() === c.title) return;
        await fetch('/api/conversations/' + c.id, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: newTitle.trim() })
        });
        loadConversationList();
      });
      item.querySelector('.del-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        await fetch('/api/conversations/' + c.id, { method: 'DELETE' });
        if (c.id === activeConvId) startNewChat();
        else loadConversationList();
      });
      convListEl.appendChild(item);
    });
    return convs;
  } catch { return []; }
}

async function openConversation(convId) {
  activeConvId = convId;
  try {
    const r = await fetch('/api/conversations/' + convId);
    if (!r.ok) return;
    const d = await r.json();
    messagesEl.innerHTML = '';
    (d.messages || []).forEach(m => addMessage(m.role, m.text, m.attachment));
    loadConversationList();
  } catch {}
  if (isMobile()) closeSidebar();
}

function startNewChat() {
  activeConvId = null;
  messagesEl.innerHTML = '';
  showEmptyState();
  loadConversationList();
}

// --- Send / regenerate / stop ---
let isGenerating = false;
let currentAbortController = null;

function setGenerating(state) {
  isGenerating = state;
  sendBtn.classList.toggle('generating', state);
  sendBtn.title = state ? 'Stop generating' : 'Send';
  sendBtn.innerHTML = state
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>'
    : '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>';
}

async function streamReply({ message = null, attachment = null, regenerate = false } = {}) {
  showTyping();
  setGenerating(true);
  currentAbortController = new AbortController();

  // Check if user wants an image (only on fresh sends, not regenerate)
  if (!regenerate) {
    const wantsImage = IMAGE_KEYWORDS.test(message || '') && !attachment;
    if (wantsImage) {
      hideTyping();
      const generated = await tryGenerateImage(message);
      if (generated) { setGenerating(false); loadConversationList(); return; }
      showTyping();
    }
  }

  let aiTextNode = null;
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: currentAbortController.signal,
      body: JSON.stringify({
        message: message || '',
        conversation_id: activeConvId,
        attachment,
        user_name: getUserName(),
        regenerate: !!regenerate,
        model: selectedModel,
        user_api_key: (() => { const k = getActiveKey(); return k ? { provider: k.provider, key: k.value } : null; })(),
      })
    });
    if (!r.ok || !r.body) {
      hideTyping();
      addMessage('error', 'Something went wrong. Try again.');
      return;
    }
    hideTyping();
    aiTextNode = addMessage('ai', '');

    const convId = r.headers.get('X-Conversation-Id');
    if (convId) activeConvId = convId;

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;
      aiTextNode.textContent = fullText;
      scrollToBottom();
    }
    speak(fullText);
    loadConversationList();
  } catch (err) {
    hideTyping();
    if (err.name === 'AbortError') {
      // User hit stop — keep whatever text streamed in so far, just mark it as stopped.
      if (aiTextNode && !aiTextNode.textContent.trim()) aiTextNode.textContent = '[Stopped]';
    } else {
      addMessage('error', 'Network error: ' + err.message);
    }
  } finally {
    setGenerating(false);
    currentAbortController = null;
  }
}

function regenerateLast(row) {
  if (isGenerating) return;
  row.remove();
  streamReply({ regenerate: true });
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (isGenerating) return;
  const text = input.value.trim();
  if (!text && !pendingFile) return;
  const attachment = pendingFile;
  pendingFile = null;
  fileInput.value = ''; cameraInput.value = '';
  pendingAttach.classList.remove('show');
  addMessage('user', text, attachment);
  input.value = '';
  input.style.height = 'auto';
  const tonePrefix = getTonePrefix();
  streamReply({ message: tonePrefix + text, attachment });
});

sendBtn.addEventListener('click', (e) => {
  if (isGenerating) {
    e.preventDefault();
    if (currentAbortController) currentAbortController.abort();
  }
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
function autoResize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
}
input.addEventListener('input', autoResize);

const sidebarOverlay = document.getElementById('sidebar-overlay');

function isMobile() { return window.innerWidth <= 768; }

function openSidebar() {
  sidebar.classList.remove('hidden');
  if (isMobile()) sidebarOverlay.style.display = 'block';
}
function closeSidebar() {
  sidebar.classList.add('hidden');
  sidebarOverlay.style.display = 'none';
}
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.contains('hidden') ? openSidebar() : closeSidebar();
});
sidebarOverlay.addEventListener('click', closeSidebar);

// Fullscreen toggle (works on Android/desktop; iOS Safari has no real Fullscreen API,
// so it falls back to a "pseudo-fullscreen" mode that maximizes the app view instead)
const fullscreenIcon  = document.getElementById('fullscreen-icon');
const fsSupported = !!(document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen);

function isFullscreen() {
  return !!(document.fullscreenElement || document.webkitFullscreenElement) ||
    document.body.classList.contains('pseudo-fullscreen');
}
function updateFullscreenBtn() {
  if (isFullscreen()) {
    fullscreenIcon.textContent = '⤢';
    fullscreenBtn.title = 'Exit fullscreen';
    fullscreenBtn.classList.add('active');
  } else {
    fullscreenIcon.textContent = '⛶';
    fullscreenBtn.title = 'Fullscreen';
    fullscreenBtn.classList.remove('active');
  }
}
async function toggleFullscreen() {
  const el = document.documentElement;
  try {
    if (fsSupported) {
      if (!document.fullscreenElement && !document.webkitFullscreenElement) {
        if (el.requestFullscreen) await el.requestFullscreen();
        else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
      } else {
        if (document.exitFullscreen) await document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      }
    } else {
      // iOS Safari / in-app browsers: real Fullscreen API isn't available,
      // so just maximize the app view (hides scroll bounce, fills the screen).
      document.body.classList.toggle('pseudo-fullscreen');
      updateFullscreenBtn();
    }
  } catch (err) {
    console.warn('Fullscreen request failed:', err);
    // Even on failure, fall back to pseudo-fullscreen so the button still does something
    document.body.classList.toggle('pseudo-fullscreen');
    updateFullscreenBtn();
  }
}
fullscreenBtn.addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', updateFullscreenBtn);
document.addEventListener('webkitfullscreenchange', updateFullscreenBtn);

// "What should Ꮇʏᴛʜɪᴄ ᴀɪ call you?" — stored locally, sent with every chat request
function getUserName() { return localStorage.getItem('mythic_user_name') || ''; }
function setUserName(name) {
  if (name) localStorage.setItem('mythic_user_name', name);
  else localStorage.removeItem('mythic_user_name');
}
function openNameModal() {
  nameInput.value = getUserName();
  nameModalOverlay.classList.add('show');
  setTimeout(() => nameInput.focus(), 50);
}
function closeNameModal() { nameModalOverlay.classList.remove('show'); }
nameBtn.addEventListener('click', openNameModal);
nameCancelBtn.addEventListener('click', closeNameModal);
nameModalOverlay.addEventListener('click', (e) => { if (e.target === nameModalOverlay) closeNameModal(); });
nameSaveBtn.addEventListener('click', () => {
  setUserName(nameInput.value.trim());
  closeNameModal();
});
nameInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); nameSaveBtn.click(); }
  else if (e.key === 'Escape') closeNameModal();
});
// First-time visitors get a gentle one-time prompt
if (!localStorage.getItem('mythic_name_prompted')) {
  localStorage.setItem('mythic_name_prompted', '1');
  setTimeout(openNameModal, 600);
}

// Hide sidebar by default on mobile
if (isMobile()) sidebar.classList.add('hidden');
newChatBtn.addEventListener('click', startNewChat);
clearBtn.addEventListener('click', async () => {
  if (!activeConvId) return;
  await fetch('/api/conversations/' + activeConvId, { method: 'DELETE' });
  startNewChat();
});

exportBtn.addEventListener('click', async () => {
  if (!activeConvId) { alert('Start or open a chat first.'); return; }
  try {
    const r = await fetch('/api/conversations/' + activeConvId);
    if (!r.ok) return;
    const d = await r.json();
    const lines = [`# ${d.title || 'Ꮇʏᴛʜɪᴄ ᴀɪ chat'}`, ''];
    (d.messages || []).forEach(m => {
      lines.push(m.role === 'user' ? 'You:' : 'Ꮇʏᴛʜɪᴄ ᴀɪ:');
      lines.push(m.text || (m.attachment ? `[attachment: ${m.attachment.name}]` : ''));
      lines.push('');
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (d.title || 'chat').replace(/[^a-z0-9_ -]/gi, '').trim().slice(0, 60) + '.txt';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert('Export failed: ' + err.message);
  }
});

// Initial load
// ─── SETTINGS ────────────────────────────────────────────────────────────────
const settingsBtn        = document.getElementById('settings-btn');
const settingsModalOverlay=document.getElementById('settings-modal-overlay');
const settingsCloseBtn   = document.getElementById('settings-close-btn');
const accentColorInput   = document.getElementById('accent-color-input');
const fontSizeSlider     = document.getElementById('font-size-slider');
const fontSizeLabel      = document.getElementById('font-size-label');
const toneSelect         = document.getElementById('tone-select');
const lengthSelect       = document.getElementById('length-select');
const customInstructions = document.getElementById('custom-instructions-input');

// Load saved settings
function loadSettings() {
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  // Theme
  const theme = s.theme || 'dark';
  applyTheme(theme);
  document.querySelectorAll('[data-group="theme"]').forEach(b => {
    b.style.borderColor = b.dataset.value === theme ? 'var(--accent)' : 'var(--border)';
    b.style.color = b.dataset.value === theme ? 'var(--accent)' : '';
  });
  // Accent
  const accent = s.accent || '#10a37f';
  accentColorInput.value = accent;
  document.documentElement.style.setProperty('--accent', accent);
  // Font size
  const fs = s.fontSize || '14.5';
  fontSizeSlider.value = fs;
  fontSizeLabel.textContent = fs + 'px';
  document.documentElement.style.setProperty('--msg-font-size', fs + 'px');
  // Bubble style
  const bubble = s.bubble || 'comfortable';
  document.body.classList.remove('bubble-compact','bubble-comfortable','bubble-spacious');
  document.body.classList.add('bubble-' + bubble);
  document.querySelectorAll('[data-group="bubble"]').forEach(b => {
    b.style.borderColor = b.dataset.value === bubble ? 'var(--accent)' : 'var(--border)';
    b.style.color = b.dataset.value === bubble ? 'var(--accent)' : '';
  });
  // Tone & Length
  if (toneSelect) toneSelect.value = s.tone || 'default';
  if (lengthSelect) lengthSelect.value = s.length || 'default';
  // Custom instructions
  if (customInstructions) customInstructions.value = s.customInstructions || '';
}

function saveSettings() {
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  s.theme = document.body.classList.contains('theme-light') ? 'light' : 'dark';
  s.accent = accentColorInput.value;
  s.fontSize = fontSizeSlider.value;
  const bubs = ['compact','comfortable','spacious'].find(b => document.body.classList.contains('bubble-'+b)) || 'comfortable';
  s.bubble = bubs;
  s.tone = toneSelect ? toneSelect.value : 'default';
  s.length = lengthSelect ? lengthSelect.value : 'default';
  s.customInstructions = customInstructions ? customInstructions.value : '';
  localStorage.setItem('mythic_settings', JSON.stringify(s));
}

function applyTheme(t) {
  if (t === 'system') {
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.body.classList.toggle('theme-light', !dark);
  } else {
    document.body.classList.toggle('theme-light', t === 'light');
  }
}

settingsBtn.addEventListener('click', () => { settingsModalOverlay.style.display = 'flex'; });
settingsCloseBtn.addEventListener('click', () => { saveSettings(); settingsModalOverlay.style.display = 'none'; });
settingsModalOverlay.addEventListener('click', e => { if (e.target === settingsModalOverlay) { saveSettings(); settingsModalOverlay.style.display = 'none'; } });

document.querySelectorAll('.settings-choice').forEach(btn => {
  btn.addEventListener('click', () => {
    const group = btn.dataset.group;
    document.querySelectorAll(`[data-group="${group}"]`).forEach(b => {
      b.style.borderColor = 'var(--border)'; b.style.color = '';
    });
    btn.style.borderColor = 'var(--accent)'; btn.style.color = 'var(--accent)';
    if (group === 'theme') applyTheme(btn.dataset.value);
    if (group === 'bubble') {
      document.body.classList.remove('bubble-compact','bubble-comfortable','bubble-spacious');
      document.body.classList.add('bubble-' + btn.dataset.value);
    }
    saveSettings();
  });
});

accentColorInput.addEventListener('input', () => {
  document.documentElement.style.setProperty('--accent', accentColorInput.value);
});
fontSizeSlider.addEventListener('input', () => {
  fontSizeLabel.textContent = fontSizeSlider.value + 'px';
  document.documentElement.style.setProperty('--msg-font-size', fontSizeSlider.value + 'px');
});

loadSettings();

// ─── PWA INSTALL BUTTON ──────────────────────────────────────────────────────
const installBtn = document.getElementById('install-btn');
let _deferredInstallPrompt = null;

function _showInstallBtn() { if (installBtn) { installBtn.style.display = 'flex'; installBtn.style.alignItems = 'center'; } }
function _hideInstallBtn() { if (installBtn) installBtn.style.display = 'none'; }

window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault(); _deferredInstallPrompt = e; _showInstallBtn();
});
window.addEventListener('appinstalled', () => {
  _hideInstallBtn(); _deferredInstallPrompt = null;
  localStorage.setItem('mythic_pwa_installed', '1');
});
if (window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone) {
  _hideInstallBtn();
} else if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone) {
  if (!localStorage.getItem('mythic_pwa_installed')) _showInstallBtn();
}

function _showIOSInstallModal() {
  const ex = document.getElementById('ios-install-modal');
  if (ex) { ex.style.display = 'flex'; return; }
  const m = document.createElement('div');
  m.id = 'ios-install-modal';
  m.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:9999;display:flex;align-items:flex-end;justify-content:center;padding:20px;';
  m.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:20px;padding:28px 24px;width:100%;max-width:420px;text-align:center;">
    <div style="font-size:42px;margin-bottom:10px;">📲</div>
    <div style="font-weight:700;font-size:18px;margin-bottom:8px;">Install Ꮇʏᴛʜɪᴄ ᴀɪ</div>
    <div style="color:var(--muted);font-size:13.5px;line-height:1.7;margin-bottom:20px;">
      Tap the <strong>Share button</strong> ⬆ at the bottom of Safari,<br>
      then tap <strong>"Add to Home Screen"</strong> ➕
    </div>
    <button id="ios-close" style="background:var(--accent);color:#fff;border:none;border-radius:10px;padding:12px 32px;font-size:15px;font-weight:600;cursor:pointer;font-family:inherit;width:100%;">Got it!</button>
  </div>`;
  document.body.appendChild(m);
  m.addEventListener('click', e => { if (e.target === m) m.remove(); });
  document.getElementById('ios-close').addEventListener('click', () => m.remove());
}

if (installBtn) installBtn.addEventListener('click', async () => {
  if (_deferredInstallPrompt) {
    _deferredInstallPrompt.prompt();
    const { outcome } = await _deferredInstallPrompt.userChoice;
    if (outcome === 'accepted') { _hideInstallBtn(); _deferredInstallPrompt = null; }
  } else if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone) {
    _showIOSInstallModal();
  } else if (window.matchMedia('(display-mode: standalone)').matches) {
    _hideInstallBtn();
  } else {
    alert('Install Ꮇʏᴛʜɪᴄ ᴀɪ as an app:\n\n• Chrome/Edge: Click ⋮ → "Install app"\n• Samsung Browser: Tap ⋮ → "Add page to"\n• Safari (iOS): Tap Share ⬆ → "Add to Home Screen"');
  }
});

// ─── NOTIFICATION BANNER ─────────────────────────────────────────────────────
const notifBanner    = document.getElementById('notif-banner');
const notifAllowBtn  = document.getElementById('notif-banner-allow');
const notifDismissBtn= document.getElementById('notif-banner-dismiss');
let _swReg = null;

function _hideBanner() { if (notifBanner) notifBanner.style.display = 'none'; }
function _showBanner() {
  if (!notifBanner || !('Notification' in window)) return;
  if (Notification.permission !== 'default') return;
  const ts = parseInt(localStorage.getItem('mythic_notif_dismissed') || '0', 10);
  if (ts && Date.now() < ts) return;
  if (ts) localStorage.removeItem('mythic_notif_dismissed');
  notifBanner.style.display = 'flex';
}
function _urlB64(b64url) {
  const pad = '='.repeat((4 - b64url.length % 4) % 4);
  return Uint8Array.from(atob((b64url + pad).replace(/-/g,'+').replace(/_/g,'/')), c => c.charCodeAt(0));
}
async function _doSubscribe(reg) {
  if (!('PushManager' in window)) return;
  try {
    const kr = await fetch('/api/push/vapid-public-key');
    if (!kr.ok) return;
    const { publicKey } = await kr.json();
    if (!publicKey) return;
    const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: _urlB64(publicKey) });
    await fetch('/api/push/subscribe', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ subscription: sub.toJSON() }) });
    localStorage.setItem('mythic_push_subscribed', '1');
  } catch(err) { console.warn('[Push]', err); }
}
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js', { scope: '/' })
    .then(reg => { _swReg = reg; if (Notification.permission === 'granted') { _hideBanner(); _doSubscribe(reg); } else _showBanner(); })
    .catch(() => _showBanner());
} else { _showBanner(); }

if (notifAllowBtn) notifAllowBtn.addEventListener('click', async () => {
  _hideBanner();
  let perm; try { perm = await Notification.requestPermission(); } catch { perm = 'denied'; }
  if (perm === 'granted') {
    if (_swReg) { try { await _swReg.showNotification('Ꮇʏᴛʜɪᴄ ᴀɪ 🔔', { body: "Notifications on! I'll let you know when your answer is ready.", icon: '/icon.png', badge: '/icon.png', tag: 'notif-on', vibrate: [150, 80, 150] }); } catch {} _doSubscribe(_swReg); }
    else { try { new Notification('Ꮇʏᴛʜɪᴄ ᴀɪ 🔔', { body: 'Notifications enabled!', icon: '/icon.png' }); } catch {} }
    const nb = document.getElementById('notif-toggle-btn'), ns = document.getElementById('notif-status');
    if (nb) { nb.textContent = 'Enabled ✓'; nb.style.borderColor = 'var(--accent)'; nb.style.color = 'var(--accent)'; }
    if (ns) ns.textContent = "You'll get notified when Ꮇʏᴛʜɪᴄ ᴀɪ replies while you're away.";
    localStorage.removeItem('mythic_notif_dismissed');
  }
});
if (notifDismissBtn) notifDismissBtn.addEventListener('click', () => {
  _hideBanner();
  localStorage.setItem('mythic_notif_dismissed', String(Date.now() + 3 * 24 * 60 * 60 * 1000));
});
window._notifyAiReply = function(preview) {
  if (document.visibilityState === 'visible' || Notification.permission !== 'granted') return;
  const body = preview || 'Your answer is ready — tap to read it.';
  const opts = { body, icon: '/icon.png', badge: '/icon.png', tag: 'mythic-reply', renotify: true, vibrate: [200, 100, 200], data: { url: '/' }, actions: [{ action: 'open', title: '💬 Open Chat' }, { action: 'dismiss', title: '✕' }] };
  if (_swReg) { try { _swReg.showNotification('Ꮇʏᴛʜɪᴄ ᴀɪ replied 💬', opts); } catch {} }
  else { try { new Notification('Ꮇʏᴛʜɪᴄ ᴀɪ replied 💬', { body, icon: '/icon.png' }); } catch {} }
};

// Notification settings toggle in Settings panel
(function() {
  const notifBtn = document.getElementById('notif-toggle-btn');
  const notifStatus = document.getElementById('notif-status');
  if (!notifBtn) return;
  function updateNotifUI() {
    if (!('Notification' in window)) { notifBtn.textContent = 'Not supported'; notifBtn.disabled = true; return; }
    if (Notification.permission === 'granted') { notifBtn.textContent = 'Enabled ✓'; notifBtn.style.borderColor = 'var(--accent)'; notifBtn.style.color = 'var(--accent)'; if (notifStatus) notifStatus.textContent = "You'll get notified when Ꮇʏᴛʜɪᴄ ᴀɪ replies while you're away."; }
    else if (Notification.permission === 'denied') { notifBtn.textContent = 'Blocked'; notifBtn.style.borderColor = '#ef4444'; notifBtn.style.color = '#ef4444'; if (notifStatus) notifStatus.textContent = 'Blocked. Allow in browser site settings.'; }
    else { notifBtn.textContent = 'Enable'; notifBtn.style.borderColor = 'var(--border)'; notifBtn.style.color = 'var(--muted)'; if (notifStatus) notifStatus.textContent = "Get notified when Ꮇʏᴛʜɪᴄ ᴀɪ replies while you're in another tab."; }
  }
  updateNotifUI();
  if (settingsBtn) settingsBtn.addEventListener('click', updateNotifUI);
  notifBtn.addEventListener('click', async () => {
    if (Notification.permission === 'denied') { if (notifStatus) notifStatus.textContent = 'Allow in browser site settings, then reload.'; return; }
    if (Notification.permission === 'granted') {
      try { const reg = await navigator.serviceWorker.getRegistration('/'); if (reg) { const sub = await reg.pushManager.getSubscription(); if (sub) { await fetch('/api/push/unsubscribe', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ endpoint: sub.endpoint }) }); await sub.unsubscribe(); localStorage.removeItem('mythic_push_subscribed'); } } } catch {}
      if (notifStatus) notifStatus.textContent = 'Notifications disabled.'; updateNotifUI(); return;
    }
    const perm = await Notification.requestPermission();
    if (perm === 'granted') { try { const reg = await navigator.serviceWorker.getRegistration('/'); if (reg) await _doSubscribe(reg); } catch {} }
    updateNotifUI();
  });
})();

// ─── API KEY MANAGER ─────────────────────────────────────────────────────────
// Keys are stored locally in localStorage (never sent to the server in plaintext
// for storage — only used as the Authorization header in the browser's own fetch
// calls to Groq/Cerebras when a key is active). The active key is injected by
// the frontend into each /api/chat request as a "user_api_key" field; the
// backend uses it instead of the server-side key when provided.
const API_KEY_STORE_KEY = 'mythic_api_keys';     // localStorage key for saved keys
const API_KEY_ACTIVE_KEY = 'mythic_api_key_active'; // localStorage key for active key id

function loadApiKeys() { try { return JSON.parse(localStorage.getItem(API_KEY_STORE_KEY) || '[]'); } catch { return []; } }
function saveApiKeys(keys) { localStorage.setItem(API_KEY_STORE_KEY, JSON.stringify(keys)); }
function getActiveKeyId() { return localStorage.getItem(API_KEY_ACTIVE_KEY) || null; }
function setActiveKeyId(id) { if (id) localStorage.setItem(API_KEY_ACTIVE_KEY, id); else localStorage.removeItem(API_KEY_ACTIVE_KEY); }
function getActiveKey() { const id = getActiveKeyId(); if (!id) return null; return loadApiKeys().find(k => k.id === id) || null; }

function renderApiKeys() {
  const list = document.getElementById('api-key-list');
  if (!list) return;
  const keys = loadApiKeys();
  const activeId = getActiveKeyId();
  list.innerHTML = '';
  if (!keys.length) {
    list.innerHTML = '<div style="font-size:12px;color:var(--muted);padding:4px 0;">No keys saved yet. Add one below.</div>';
    return;
  }
  keys.forEach(k => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:6px;padding:7px 10px;border-radius:8px;border:1.5px solid ' + (k.id === activeId ? 'var(--accent)' : 'var(--border)') + ';background:' + (k.id === activeId ? 'var(--accent-dim)' : 'var(--panel)') + ';';
    const badge = document.createElement('span');
    badge.style.cssText = 'font-size:10px;font-weight:700;padding:2px 6px;border-radius:6px;background:' + (k.provider === 'groq' ? '#f59e0b22' : '#8b5cf622') + ';color:' + (k.provider === 'groq' ? '#f59e0b' : '#8b5cf6') + ';flex-shrink:0;';
    badge.textContent = k.provider.toUpperCase();
    const label = document.createElement('span');
    label.style.cssText = 'flex:1;font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' + (k.id === activeId ? 'color:var(--accent);font-weight:600;' : '');
    label.textContent = (k.label || 'Unnamed') + '  •••' + k.value.slice(-6);
    const useBtn = document.createElement('button');
    useBtn.type = 'button';
    useBtn.style.cssText = 'font-size:11px;padding:3px 10px;border-radius:6px;cursor:pointer;font-family:inherit;border:1px solid ' + (k.id === activeId ? 'var(--accent)' : 'var(--border)') + ';background:' + (k.id === activeId ? 'var(--accent)' : 'none') + ';color:' + (k.id === activeId ? '#fff' : 'var(--muted)') + ';white-space:nowrap;';
    useBtn.textContent = k.id === activeId ? '✓ Active' : 'Use';
    useBtn.addEventListener('click', () => { setActiveKeyId(k.id === activeId ? null : k.id); renderApiKeys(); });
    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.style.cssText = 'font-size:12px;padding:3px 8px;border-radius:6px;cursor:pointer;font-family:inherit;border:1px solid var(--border);background:none;color:#ef4444;';
    delBtn.textContent = '✕';
    delBtn.title = 'Delete this key';
    delBtn.addEventListener('click', () => {
      const updated = loadApiKeys().filter(x => x.id !== k.id);
      saveApiKeys(updated);
      if (getActiveKeyId() === k.id) setActiveKeyId(null);
      renderApiKeys();
    });
    row.appendChild(badge); row.appendChild(label); row.appendChild(useBtn); row.appendChild(delBtn);
    list.appendChild(row);
  });
}

// Re-render when settings opens
if (settingsBtn) settingsBtn.addEventListener('click', renderApiKeys);

const apiKeyAddBtn = document.getElementById('api-key-add-btn');
const apiKeyError  = document.getElementById('api-key-error');
if (apiKeyAddBtn) apiKeyAddBtn.addEventListener('click', () => {
  const provider = document.getElementById('api-key-provider-select').value;
  const label    = document.getElementById('api-key-label-input').value.trim();
  const value    = document.getElementById('api-key-value-input').value.trim();
  if (apiKeyError) apiKeyError.style.display = 'none';
  if (!value) { if (apiKeyError) { apiKeyError.textContent = 'Paste your API key first.'; apiKeyError.style.display = 'block'; } return; }
  if (value.length < 20) { if (apiKeyError) { apiKeyError.textContent = 'That doesn\'t look like a valid API key.'; apiKeyError.style.display = 'block'; } return; }
  const keys = loadApiKeys();
  if (keys.some(k => k.value === value)) { if (apiKeyError) { apiKeyError.textContent = 'This key is already saved.'; apiKeyError.style.display = 'block'; } return; }
  keys.push({ id: 'k_' + Date.now(), provider, label: label || (provider + ' key'), value });
  saveApiKeys(keys);
  document.getElementById('api-key-label-input').value = '';
  document.getElementById('api-key-value-input').value = '';
  renderApiKeys();
});

// ─── DOWNLOADABLE FILE GENERATION FROM CHAT ──────────────────────────────────
// Detects when AI says "here is your [file type]" or includes a code block,
// and offers a Download button. Also handles explicit /api/generate-file requests.
function _triggerDownload(b64, filename, mimeType) {
  const a = document.createElement('a');
  a.href = 'data:' + mimeType + ';base64,' + b64;
  a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
}

// After each AI message, scan for downloadable content
function _addDownloadButtonIfNeeded(row, textContent) {
  if (!textContent || textContent.length < 20) return;
  // Check if this looks like generated file content (code block or explicit file)
  const hasCodeBlock = /```[\w\s]*\n[\s\S]+?```/.test(textContent);
  const isFileMention = /\b(pdf|docx|\.txt|\.csv|\.json|\.html|\.css|\.js|\.py|\.md|spreadsheet|document|download)\b/i.test(textContent);
  if (!hasCodeBlock && !isFileMention) return;
  if (row.querySelector('.download-file-btn')) return;

  const bar = document.createElement('div');
  bar.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;';

  // Extract code blocks for direct download
  const codeBlocks = [...textContent.matchAll(/```(\w*)\n([\s\S]*?)```/g)];
  codeBlocks.forEach((match, i) => {
    const lang = (match[1] || 'txt').toLowerCase();
    const code = match[2];
    const extMap = { python:'py', javascript:'js', js:'js', typescript:'ts', html:'html', css:'css',
      json:'json', csv:'csv', markdown:'md', md:'md', sql:'sql', bash:'sh', shell:'sh',
      yaml:'yaml', xml:'xml', r:'r', cpp:'cpp', c:'c', java:'java', rust:'rs' };
    const ext = extMap[lang] || lang || 'txt';
    const mimeMap = { html:'text/html', css:'text/css', js:'text/javascript', json:'application/json',
      csv:'text/csv', py:'text/x-python', md:'text/markdown', sql:'text/plain', sh:'text/plain' };
    const mime = mimeMap[ext] || 'text/plain';
    const btn = document.createElement('button');
    btn.className = 'download-file-btn';
    btn.style.cssText = 'background:var(--panel);border:1px solid var(--accent);color:var(--accent);font-size:11.5px;padding:5px 11px;border-radius:8px;cursor:pointer;font-family:inherit;';
    btn.textContent = '⬇ Download .' + ext + (codeBlocks.length > 1 ? ' (' + (i+1) + ')' : '');
    btn.addEventListener('click', () => {
      const b64 = btoa(unescape(encodeURIComponent(code)));
      _triggerDownload(b64, 'mythic-ai-output-' + (i+1) + '.' + ext, mime);
    });
    bar.appendChild(btn);
  });

  // For non-code-block content, offer full message as TXT
  if (!codeBlocks.length && isFileMention) {
    const btn = document.createElement('button');
    btn.className = 'download-file-btn';
    btn.style.cssText = 'background:var(--panel);border:1px solid var(--border);color:var(--muted);font-size:11.5px;padding:5px 11px;border-radius:8px;cursor:pointer;font-family:inherit;';
    btn.textContent = '⬇ Save as .txt';
    btn.addEventListener('click', () => {
      const b64 = btoa(unescape(encodeURIComponent(textContent)));
      _triggerDownload(b64, 'mythic-ai-reply.txt', 'text/plain');
    });
    bar.appendChild(btn);
  }

  // PDF download via backend (if text content is substantial)
  if (textContent.length > 100) {
    const pdfBtn = document.createElement('button');
    pdfBtn.className = 'download-file-btn';
    pdfBtn.style.cssText = 'background:var(--panel);border:1px solid var(--border);color:var(--muted);font-size:11.5px;padding:5px 11px;border-radius:8px;cursor:pointer;font-family:inherit;';
    pdfBtn.textContent = '⬇ Save as PDF';
    pdfBtn.addEventListener('click', async () => {
      pdfBtn.textContent = 'Generating...'; pdfBtn.disabled = true;
      try {
        const r = await fetch('/api/generate-file', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: textContent, format: 'pdf', title: 'Mythic AI Output' })
        });
        const d = await r.json();
        if (d.file) _triggerDownload(d.file, d.filename || 'mythic-ai.pdf', d.mimeType || 'application/pdf');
        else { pdfBtn.textContent = '⬇ Save as PDF'; pdfBtn.disabled = false; }
      } catch { pdfBtn.textContent = '⬇ Save as PDF'; pdfBtn.disabled = false; }
    });
    bar.appendChild(pdfBtn);
  }

  if (bar.children.length) row.appendChild(bar);
}

// Hook into the post-AI-reply processing already done by the submit listener

// ─── MARKDOWN RENDERING ──────────────────────────────────────────────────────
function renderMarkdown(text) {
  const div = document.createElement('div');
  div.className = 'msg-text md-rendered';
  // Escape HTML first
  let html = text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // Code blocks (must come before inline code)
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`);
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3 style="font-size:14px;margin:6px 0 3px;font-weight:700;">$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2 style="font-size:15px;margin:8px 0 4px;font-weight:700;">$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1 style="font-size:17px;margin:10px 0 5px;font-weight:700;">$1</h1>');
  // Unordered lists
  html = html.replace(/(^|\n)([\-\*] .+(\n[\-\*] .+)*)/g, (_, pre, block) =>
    pre + '<ul>' + block.replace(/[\-\*] (.+)/g, '<li>$1</li>') + '</ul>');
  // Ordered lists
  html = html.replace(/(^|\n)(\d+\. .+(\n\d+\. .+)*)/g, (_, pre, block) =>
    pre + '<ol>' + block.replace(/\d+\. (.+)/g, '<li>$1</li>') + '</ol>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  div.innerHTML = html;
  return div;
}

// Override addMessage to use markdown for AI
const _origAddMessage = addMessage;
function addMessage(role, text, attachment) {
  const textNode = _origAddMessage(role, text, attachment);
  if (role === 'ai' && text) {
    try {
      const md = renderMarkdown(text);
      textNode.parentNode.replaceChild(md, textNode);
      return md;
    } catch { return textNode; }
  }
  return textNode;
}

// ─── MESSAGE TIMESTAMPS ───────────────────────────────────────────────────────
const _origAddMsg2 = addMessage;
function addMessageWithTimestamp(role, text, attachment) {
  const node = _origAddMsg2(role, text, attachment);
  const row = node.closest ? node.closest('.msg-row') : null;
  if (row) {
    const ts = document.createElement('div');
    ts.className = 'msg-timestamp';
    ts.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    row.appendChild(ts);
  }
  return node;
}
// Monkey-patch addMessage to include timestamp
const _rawAdd = addMessage;
window._addMsgFinal = function(role, text, attachment) {
  const node = _rawAdd(role, text, attachment);
  const row = node && node.closest ? node.closest('.msg-row') : null;
  if (row && !row.querySelector('.msg-timestamp')) {
    const ts = document.createElement('div');
    ts.className = 'msg-timestamp';
    ts.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    row.appendChild(ts);
  }
  return node;
};

// ─── FOLLOW-UP SUGGESTIONS ───────────────────────────────────────────────────
async function addFollowupSuggestions(aiText) {
  if (!aiText || aiText.length < 50) return;
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        message: `Based on this AI reply, suggest 3 short follow-up questions the user might ask. Reply ONLY with 3 questions, one per line, no numbering, no extra text:\n\n${aiText.slice(0,400)}`,
        conversation_id: null, model: 'mythic-1.0', user_name: ''
      })
    });
    if (!r.ok) return;
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let full = '';
    while (true) { const {done,value} = await reader.read(); if(done) break; full += dec.decode(value,{stream:true}); }
    const qs = full.trim().split('\n').filter(q => q.trim().length > 5).slice(0,3);
    if (!qs.length) return;
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;max-width:100%;';
    qs.forEach(q => {
      const btn = document.createElement('button');
      btn.textContent = q.trim().replace(/^["']|["']$/g,'');
      btn.style.cssText = 'background:var(--panel);border:1px solid var(--border);color:var(--muted);font-size:11.5px;padding:5px 10px;border-radius:16px;cursor:pointer;font-family:inherit;text-align:left;';
      btn.addEventListener('click', () => {
        input.value = btn.textContent; input.focus(); autoResize();
        form.requestSubmit(); wrap.remove();
      });
      wrap.appendChild(btn);
    });
    messagesEl.appendChild(wrap);
    scrollToBottom();
  } catch {}
}

// ─── MESSAGE REACTIONS ────────────────────────────────────────────────────────
function addReactionBar(row) {
  if (row.querySelector('.reaction-bar')) return;
  const bar = document.createElement('div');
  bar.className = 'reaction-bar';
  bar.style.cssText = 'display:flex;gap:4px;margin-top:3px;';
  ['👍','👎','❤️','😂','🔖'].forEach(emoji => {
    const btn = document.createElement('button');
    btn.textContent = emoji;
    btn.style.cssText = 'background:none;border:1px solid var(--border);border-radius:12px;padding:2px 7px;font-size:13px;cursor:pointer;touch-action:manipulation;';
    btn.addEventListener('click', () => {
      btn.style.borderColor = btn.style.borderColor === 'var(--accent)' ? 'var(--border)' : 'var(--accent)';
      btn.style.background  = btn.style.background === 'var(--accent-dim)' ? '' : 'var(--accent-dim)';
    });
    bar.appendChild(btn);
  });
  row.appendChild(bar);
}

// ─── MESSAGE SEARCH ───────────────────────────────────────────────────────────
function addSearchUI() {
  const searchWrap = document.createElement('div');
  searchWrap.id = 'msg-search-wrap';
  searchWrap.style.cssText = 'display:none;position:fixed;top:70px;left:50%;transform:translateX(-50%);z-index:150;background:var(--panel);border:1.5px solid var(--accent);border-radius:10px;padding:8px 12px;display:flex;gap:8px;align-items:center;min-width:280px;box-shadow:0 4px 20px rgba(0,0,0,.3);';
  searchWrap.innerHTML = '<input id="msg-search-input" placeholder="Search messages..." style="background:transparent;border:none;color:var(--text);font-size:14px;outline:none;flex:1;font-family:inherit;"><span id="msg-search-count" style="color:var(--muted);font-size:12px;"></span><button id="msg-search-close" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;">✕</button>';
  searchWrap.style.display = 'none';
  document.body.appendChild(searchWrap);

  let highlights = [];
  document.getElementById('msg-search-input').addEventListener('input', e => {
    highlights.forEach(el => { el.style.background = ''; el.style.outline = ''; });
    highlights = [];
    const q = e.target.value.trim().toLowerCase();
    if (!q) { document.getElementById('msg-search-count').textContent = ''; return; }
    document.querySelectorAll('.msg-text,.md-rendered').forEach(el => {
      if (el.textContent.toLowerCase().includes(q)) {
        el.style.background = 'rgba(255,200,0,.15)';
        el.style.outline = '2px solid rgba(255,200,0,.4)';
        highlights.push(el);
      }
    });
    document.getElementById('msg-search-count').textContent = highlights.length ? `${highlights.length} found` : 'no results';
    if (highlights.length) highlights[0].scrollIntoView({behavior:'smooth',block:'center'});
  });
  document.getElementById('msg-search-close').addEventListener('click', () => {
    searchWrap.style.display = 'none';
    highlights.forEach(el => { el.style.background = ''; el.style.outline = ''; });
  });
  return searchWrap;
}
const msgSearchWrap = addSearchUI();

// Keyboard shortcut Ctrl+F to search messages
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'f' && !settingsModalOverlay.style.display.includes('flex')) {
    e.preventDefault();
    msgSearchWrap.style.display = 'flex';
    setTimeout(() => document.getElementById('msg-search-input').focus(), 50);
  }
  if (e.key === 'Escape') msgSearchWrap.style.display = 'none';
});

// ─── PWA SUPPORT ─────────────────────────────────────────────────────────────
if ('serviceWorker' in navigator && navigator.serviceWorker) {
  // Inline service worker for offline caching
  const swCode = `
const CACHE = 'mythic-ai-v1';
const OFFLINE = ['/', '/static/app.js'];
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(['/']))));
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});`;
  const blob = new Blob([swCode], {type:'application/javascript'});
  navigator.serviceWorker.register(URL.createObjectURL(blob)).catch(()=>{});
}

// ─── WIRE REACTIONS INTO MSG ACTIONS ─────────────────────────────────────────
// Patch buildMsgActions to add reaction button
const _origBuildActions = buildMsgActions;
function buildMsgActions(row, textNode, role) {
  const actions = _origBuildActions(row, textNode, role);
  if (role === 'ai') {
    const reactBtn = document.createElement('button');
    reactBtn.type = 'button'; reactBtn.title = 'React'; reactBtn.textContent = '😊';
    reactBtn.addEventListener('click', () => addReactionBar(row));
    actions.appendChild(reactBtn);
    // 🔊 speak button
    const sp = document.createElement('button');
    sp.type='button'; sp.title='Read aloud'; sp.textContent='🔊';
    sp.addEventListener('click', () => {
      if (sp.textContent === '⏹') { stopSpeaking(); sp.textContent='🔊'; return; }
      sp.textContent='⏹'; speak(textNode.textContent || (textNode.innerText || ''));
      if (currentUtterance) {
        currentUtterance.onend = () => sp.textContent='🔊';
        currentUtterance.onerror = () => sp.textContent='🔊';
      }
    });
    actions.appendChild(sp);
  }
  return actions;
}

function stopSpeaking() { if(window.speechSynthesis) window.speechSynthesis.cancel(); }

// ─── TONE/LENGTH INJECTION ────────────────────────────────────────────────────
function getTonePrefix() {
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  const tone = s.tone || 'default';
  const length = s.length || 'default';
  let parts = [];
  if (tone === 'formal') parts.push('Reply in a formal, professional tone.');
  if (tone === 'casual') parts.push('Reply in a casual, friendly tone.');
  if (tone === 'funny') parts.push('Be funny and use wit and humor in your reply.');
  if (tone === 'professional') parts.push('Use a professional, business-appropriate tone.');
  if (length === 'short') parts.push('Keep your reply very short — 1-3 sentences max.');
  if (length === 'medium') parts.push('Keep your reply medium length — a few paragraphs.');
  if (length === 'long') parts.push('Give a thorough, detailed, long reply.');
  const ci = customInstructions ? customInstructions.value.trim() : '';
  if (ci) parts.push(ci);
  return parts.length ? '[Instructions: ' + parts.join(' ') + '] ' : '';
}

// ─── AUTO FOLLOW-UPS AFTER AI REPLY ──────────────────────────────────────────
// Hook into streamReply completion by patching the form submit
const _origFormSubmit = form.onsubmit;
form.addEventListener('submit', async () => {
  // Wait for generation to finish then add follow-ups
  const checkDone = setInterval(() => {
    if (!isGenerating) {
      clearInterval(checkDone);
      const allRows = messagesEl.querySelectorAll('.msg-row.ai');
      if (allRows.length) {
        const lastRow = allRows[allRows.length - 1];
        const textEl = lastRow.querySelector('.msg-text,.md-rendered');
        if (textEl && !lastRow.querySelector('.reaction-bar')) {
          addReactionBar(lastRow);
          const fullText = textEl.textContent || textEl.innerText || '';
          _addDownloadButtonIfNeeded(lastRow, fullText);
          setTimeout(() => addFollowupSuggestions(fullText), 300);
        }
      }
    }
  }, 500);
});

// ─── INITIAL LOAD ─────────────────────────────────────────────────────────────
(async () => {
  const convs = await loadConversationList();
  if (convs.length > 0) openConversation(convs[0].id);
  else showEmptyState();
})();

const imgGenBtn   = document.getElementById('img-gen-btn');
const ghibliBtn   = document.getElementById('ghibli-btn');
const homeworkBtn = document.getElementById('homework-btn');
const weatherBtn2 = document.getElementById('weather-btn');
const searchBtn   = document.getElementById('search-btn');

if (imgGenBtn) imgGenBtn.addEventListener('click', () => {
  const imgModal = document.getElementById('img-modal-overlay');
  if (imgModal) { imgModal.style.display = 'flex'; document.getElementById('img-prompt').focus(); }
});
if (homeworkBtn) homeworkBtn.addEventListener('click', () => {
  input.value = 'Help me with my homework: '; input.focus(); autoResize();
  input.setSelectionRange(input.value.length, input.value.length);
});
if (weatherBtn2) weatherBtn2.addEventListener('click', () => {
  const wm = document.getElementById('weather-modal-overlay');
  if (wm) { wm.style.display = 'flex'; document.getElementById('weather-city').focus(); }
});
if (searchBtn) searchBtn.addEventListener('click', () => {
  const q = prompt('What do you want to search for?');
  if (!q || !q.trim()) return;
  input.value = 'Search: ' + q.trim(); autoResize(); form.requestSubmit();
});

// ─── GHIBLI SELFIE MODAL ─────────────────────────────────────────────────────
const ghibliModal     = document.getElementById('ghibli-modal-overlay');
const ghibliUploadArea= document.getElementById('ghibli-upload-area');
const ghibliFileInput = document.getElementById('ghibli-file-input');
const ghibliPreviewWrap=document.getElementById('ghibli-preview-wrap');
const ghibliPreview   = document.getElementById('ghibli-preview');
const ghibliResult    = document.getElementById('ghibli-result');
const ghibliResultWrap= document.getElementById('ghibli-result-wrap');
const ghibliLoading   = document.getElementById('ghibli-loading');
const ghibliError     = document.getElementById('ghibli-error');
const ghibliGenerateBtn=document.getElementById('ghibli-generate-btn');
const ghibliCloseBtn  = document.getElementById('ghibli-close-btn');
const ghibliDownloadBtn=document.getElementById('ghibli-download-btn');
const ghibliExtraInput= document.getElementById('ghibli-extra');

let ghibliBase64 = null;
let ghibliMimeType = 'image/jpeg';
let ghibliSelectedStyle = 'Studio Ghibli portrait, Spirited Away style, soft watercolor anime art';

// Open/close
if (ghibliBtn) ghibliBtn.addEventListener('click', () => {
  ghibliModal.style.display = 'flex';
  ghibliBase64 = null;
  ghibliPreviewWrap.style.display = 'none';
  ghibliResultWrap.style.display = 'none';
  ghibliError.style.display = 'none';
  ghibliLoading.style.display = 'none';
});
if (ghibliCloseBtn) ghibliCloseBtn.addEventListener('click', () => ghibliModal.style.display = 'none');
ghibliModal.addEventListener('click', e => { if (e.target === ghibliModal) ghibliModal.style.display = 'none'; });

// Style selector
document.querySelectorAll('.ghibli-style-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.ghibli-style-btn').forEach(b => {
      b.style.borderColor = 'var(--border)'; b.style.background = 'var(--panel)'; b.style.color = 'var(--muted)';
    });
    btn.style.borderColor = 'var(--accent)'; btn.style.background = 'var(--accent-dim)'; btn.style.color = 'var(--accent)';
    ghibliSelectedStyle = btn.dataset.style;
  });
});

// Upload area
ghibliUploadArea.addEventListener('click', () => ghibliFileInput.click());
ghibliUploadArea.addEventListener('dragover', e => { e.preventDefault(); ghibliUploadArea.style.borderColor = 'var(--accent)'; });
ghibliUploadArea.addEventListener('dragleave', () => { ghibliUploadArea.style.borderColor = 'var(--border)'; });
ghibliUploadArea.addEventListener('drop', e => {
  e.preventDefault(); ghibliUploadArea.style.borderColor = 'var(--border)';
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadGhibliPhoto(file);
});
ghibliFileInput.addEventListener('change', () => {
  if (ghibliFileInput.files[0]) loadGhibliPhoto(ghibliFileInput.files[0]);
});

function loadGhibliPhoto(file) {
  const reader = new FileReader();
  reader.onload = e => {
    ghibliBase64 = e.target.result.split(',')[1];
    ghibliMimeType = file.type || 'image/jpeg';
    ghibliPreview.src = e.target.result;
    ghibliPreviewWrap.style.display = 'block';
    ghibliResultWrap.style.display = 'none';
    ghibliError.style.display = 'none';
    ghibliUploadArea.style.borderColor = 'var(--accent)';
  };
  reader.readAsDataURL(file);
}

// Generate Ghibli image
ghibliGenerateBtn.addEventListener('click', async () => {
  if (!ghibliBase64) {
    ghibliError.textContent = 'Please upload your photo first!';
    ghibliError.style.display = 'block'; return;
  }
  ghibliError.style.display = 'none';
  ghibliResultWrap.style.display = 'none';
  ghibliLoading.style.display = 'block';
  ghibliGenerateBtn.disabled = true;

  const extra = ghibliExtraInput.value.trim();
  const prompt = `${ghibliSelectedStyle}, beautiful detailed portrait of a person, ${extra ? extra + ', ' : ''}masterpiece, best quality, highly detailed, cinematic lighting, soft colors, dreamy atmosphere`;

  try {
    const r = await fetch('/api/generate-image', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ prompt, imageBase64: ghibliBase64, mimeType: ghibliMimeType })
    });
    const d = await r.json();
    ghibliLoading.style.display = 'none';
    if (d.image) {
      ghibliResult.src = 'data:image/png;base64,' + d.image;
      ghibliResultWrap.style.display = 'block';
      // Also show in chat
      clearEmptyState();
      const row = document.createElement('div'); row.className = 'msg-row ai';
      const bubble = document.createElement('div'); bubble.className = 'msg ai';
      bubble.style.padding = '8px';
      const cap = document.createElement('div');
      cap.textContent = '🌿 Your Ghibli portrait';
      cap.style.cssText = 'font-size:12px;color:var(--muted);margin-bottom:8px;';
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + d.image;
      img.style.cssText = 'max-width:100%;border-radius:12px;display:block;cursor:pointer;';
      img.title = 'Click to download';
      img.addEventListener('click', () => downloadGhibliImage(d.image));
      bubble.appendChild(cap); bubble.appendChild(img); row.appendChild(bubble);
      messagesEl.appendChild(row); scrollToBottom();
    } else {
      ghibliError.textContent = d.error || 'Generation failed. Try again.';
      ghibliError.style.display = 'block';
    }
  } catch (e) {
    ghibliLoading.style.display = 'none';
    ghibliError.textContent = 'Error: ' + e.message;
    ghibliError.style.display = 'block';
  } finally { ghibliGenerateBtn.disabled = false; }
});

function downloadGhibliImage(b64) {
  const a = document.createElement('a');
  a.href = 'data:image/png;base64,' + b64;
  a.download = 'mythic-ai-ghibli-portrait.png';
  document.body.appendChild(a); a.click(); a.remove();
}
if (ghibliDownloadBtn) ghibliDownloadBtn.addEventListener('click', () => {
  if (ghibliResult.src) downloadGhibliImage(ghibliResult.src.split(',')[1]);
});

// ─── IMAGE GENERATION MODAL JS ───────────────────────────────────────────────
const imgModalOverlay = document.getElementById('img-modal-overlay');
const imgPromptEl     = document.getElementById('img-prompt');
const imgStyleEl      = document.getElementById('img-style');
const imgResultEl     = document.getElementById('img-result');
const imgOutputEl     = document.getElementById('img-output');
const imgLoadingEl    = document.getElementById('img-loading');
const imgErrorEl      = document.getElementById('img-error');
const imgGenerateBtn2 = document.getElementById('img-generate-btn');
const imgCloseBtn2    = document.getElementById('img-close-btn');

if (imgGenerateBtn2) imgGenerateBtn2.addEventListener('click', async () => {
  const prompt = imgPromptEl.value.trim();
  const style = imgStyleEl ? imgStyleEl.value : '';
  if (!prompt) return;
  imgResultEl.style.display = 'none'; imgErrorEl.style.display = 'none';
  imgLoadingEl.style.display = 'block'; imgGenerateBtn2.disabled = true;
  try {
    const r = await fetch('/api/generate-image', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,style})});
    const d = await r.json();
    imgLoadingEl.style.display = 'none';
    if (d.image) {
      imgOutputEl.src = 'data:image/png;base64,' + d.image;
      imgResultEl.style.display = 'block';
      clearEmptyState();
      const row = document.createElement('div'); row.className = 'msg-row ai';
      const bubble = document.createElement('div'); bubble.className = 'msg ai'; bubble.style.padding='8px';
      const cap = document.createElement('div'); cap.textContent = '🎨 ' + prompt;
      cap.style.cssText = 'font-size:12px;opacity:.7;margin-bottom:8px;';
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + d.image;
      img.style.cssText = 'max-width:100%;border-radius:10px;display:block;';
      bubble.appendChild(cap); bubble.appendChild(img); row.appendChild(bubble);
      messagesEl.appendChild(row); scrollToBottom();
    } else { imgErrorEl.textContent = d.error || 'Failed.'; imgErrorEl.style.display='block'; }
  } catch(e) { imgLoadingEl.style.display='none'; imgErrorEl.textContent='Error: '+e.message; imgErrorEl.style.display='block'; }
  finally { imgGenerateBtn2.disabled = false; }
});
if (imgCloseBtn2) imgCloseBtn2.addEventListener('click', () => imgModalOverlay.style.display='none');
if (imgModalOverlay) imgModalOverlay.addEventListener('click', e => { if(e.target===imgModalOverlay) imgModalOverlay.style.display='none'; });
if (imgPromptEl) imgPromptEl.addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();imgGenerateBtn2.click();} });

// ─── WEATHER MODAL JS ────────────────────────────────────────────────────────
const weatherModal2    = document.getElementById('weather-modal-overlay');
const weatherCityEl2   = document.getElementById('weather-city');
const weatherResultEl2 = document.getElementById('weather-result');
const weatherContentEl2= document.getElementById('weather-content');
const weatherLoadingEl2= document.getElementById('weather-loading');
const weatherErrorEl2  = document.getElementById('weather-error');
const weatherSearchBtn2= document.getElementById('weather-search-btn');
const weatherCloseBtn2 = document.getElementById('weather-close-btn');
const weatherLocBtn2   = document.getElementById('weather-location-btn');

function renderWeather(w) {
  weatherContentEl2.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
      <div style="font-size:48px;">${w.icon}</div>
      <div><div style="font-size:18px;font-weight:700;">${w.location}</div>
      <div style="font-size:13px;color:var(--muted);">${w.condition}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
      <div style="background:var(--bg);border-radius:8px;padding:10px;">
        <div style="font-size:11px;color:var(--muted);">TEMPERATURE</div>
        <div style="font-size:22px;font-weight:700;">${w.temp}°C</div>
        <div style="font-size:11px;color:var(--muted);">Feels like ${w.feels_like}°C</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:10px;">
        <div style="font-size:11px;color:var(--muted);">HUMIDITY</div>
        <div style="font-size:22px;font-weight:700;">${w.humidity}%</div>
        <div style="font-size:11px;color:var(--muted);">Wind ${w.wind_speed} km/h</div>
      </div>
    </div>`;
  weatherResultEl2.style.display = 'block';
  input.value = `${w.icon} Weather in ${w.location}: ${w.temp}°C, ${w.condition}. Humidity: ${w.humidity}%, Wind: ${w.wind_speed} km/h.`;
  autoResize();
}

async function fetchWeatherModal(payload) {
  weatherResultEl2.style.display='none'; weatherErrorEl2.style.display='none';
  weatherLoadingEl2.style.display='block'; weatherSearchBtn2.disabled=true;
  try {
    const r = await fetch('/api/weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d = await r.json(); weatherLoadingEl2.style.display='none';
    if(d.weather) renderWeather(d.weather);
    else { weatherErrorEl2.textContent=d.error||'Could not fetch weather.'; weatherErrorEl2.style.display='block'; }
  } catch(e) { weatherLoadingEl2.style.display='none'; weatherErrorEl2.textContent='Error: '+e.message; weatherErrorEl2.style.display='block'; }
  finally { weatherSearchBtn2.disabled=false; }
}
if (weatherSearchBtn2) weatherSearchBtn2.addEventListener('click', () => { const loc=weatherCityEl2.value.trim(); if(loc) fetchWeatherModal({location:loc}); });
if (weatherCityEl2) weatherCityEl2.addEventListener('keydown', e => { if(e.key==='Enter') weatherSearchBtn2.click(); });
if (weatherCloseBtn2) weatherCloseBtn2.addEventListener('click', () => weatherModal2.style.display='none');
if (weatherModal2) weatherModal2.addEventListener('click', e => { if(e.target===weatherModal2) weatherModal2.style.display='none'; });
if (weatherLocBtn2) weatherLocBtn2.addEventListener('click', () => {
  if (!navigator.geolocation) { alert('Geolocation not supported'); return; }
  navigator.geolocation.getCurrentPosition(
    pos => fetchWeatherModal({lat:pos.coords.latitude,lon:pos.coords.longitude}),
    err => alert('Location error: '+err.message)
  );
});

</script>
</body>
</html>
"""

@app.route("/sw.js")
def service_worker_js():
    sw = r"""
const CACHE = 'mythic-ai-v3';
self.addEventListener('install', e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(['/']))); self.skipWaiting(); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET' || e.request.url.includes('/api/')) return;
  e.respondWith(fetch(e.request).then(resp => { const clone = resp.clone(); caches.open(CACHE).then(c => c.put(e.request, clone)); return resp; }).catch(() => caches.match(e.request)));
});
self.addEventListener('push', e => {
  let data = { title: 'Mythic AI', body: 'New message', icon: '/icon.png', url: '/' };
  try { if (e.data) data = { ...data, ...e.data.json() }; } catch {}
  e.waitUntil(self.registration.showNotification(data.title, { body: data.body, icon: data.icon, badge: '/icon.png', tag: 'mythic-reply', renotify: true, vibrate: [200,100,200], data: { url: data.url }, actions: [{ action:'open', title:'💬 Open Chat' }, { action:'dismiss', title:'✕' }] }));
});
self.addEventListener('notificationclick', e => {
  e.notification.close(); if (e.action === 'dismiss') return;
  const target = (e.notification.data && e.notification.data.url) || '/';
  e.waitUntil(clients.matchAll({ type:'window', includeUncontrolled:true }).then(list => { for (const c of list) { if (c.url.includes(self.location.origin) && 'focus' in c) { c.navigate(target); return c.focus(); } } if (clients.openWindow) return clients.openWindow(target); }));
});
"""
    return Response(sw, mimetype="application/javascript", headers={"Service-Worker-Allowed": "/"})


@app.route("/manifest.json")
def pwa_manifest():
    m = {
        "name": "\u13c6\u1eff\u1d1b\u043d\u1d04 \u1d00\u026a",
        "short_name": "Mythic AI",
        "description": "Smart AI assistant by Aarav Singh",
        "start_url": "/", "display": "standalone",
        "background_color": "#1a1a1a", "theme_color": "#10a37f",
        "orientation": "any", "scope": "/", "lang": "en",
        "categories": ["productivity", "utilities"],
        "icons": [
            {"src": "/icon.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
        "shortcuts": [{"name": "New Chat", "url": "/"}],
    }
    return Response(json.dumps(m), mimetype="application/manifest+json",
                    headers={"Cache-Control": "public, max-age=86400"})


def _make_mythic_icon_png(size=192):
    import struct, zlib
    W = H = size
    img = bytearray(W * H * 4)
    def sp(x, y, r, g, b, a=255):
        if 0 <= x < W and 0 <= y < H:
            i = (y*W+x)*4; img[i]=r; img[i+1]=g; img[i+2]=b; img[i+3]=a
    def fr(x0, y0, x1, y1, r, g, b):
        for y in range(max(0,y0), min(H,y1)):
            for x in range(max(0,x0), min(W,x1)): sp(x, y, r, g, b)
    def caa(cx, cy, rad, r, g, b):
        for y in range(cy-rad-1, cy+rad+2):
            for x in range(cx-rad-1, cx+rad+2):
                d = ((x-cx)**2+(y-cy)**2)**0.5; a = max(0, min(255, int((rad+0.5-d)*255)))
                if a > 0 and 0 <= x < W and 0 <= y < H:
                    i=(y*W+x)*4; bl=a/255; img[i]=int(img[i]*(1-bl)+r*bl); img[i+1]=int(img[i+1]*(1-bl)+g*bl); img[i+2]=int(img[i+2]*(1-bl)+b*bl); img[i+3]=min(255,img[i+3]+a)
    cr = size//4; fr(cr,0,W-cr,H,16,163,127); fr(0,cr,W,H-cr,16,163,127)
    for cx,cy in [(cr,cr),(W-cr,cr),(cr,H-cr),(W-cr,H-cr)]: caa(cx,cy,cr,16,163,127)
    s=size/40; pts=[(int(10*s),int(28*s)),(int(10*s),int(12*s)),(int(20*s),int(22*s)),(int(30*s),int(12*s)),(int(30*s),int(28*s))]; lw=max(2,size//14)
    def dl(x0,y0,x1,y1):
        dx,dy=x1-x0,y1-y0; steps=max(abs(dx),abs(dy),1)
        for i in range(steps+1):
            x=int(x0+dx*i/steps); y=int(y0+dy*i/steps)
            for ox in range(-lw//2,lw//2+1):
                for oy in range(-lw//2,lw//2+1): sp(x+ox,y+oy,255,255,255)
    for i in range(len(pts)-1): dl(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1])
    def chunk(name, data): crc=zlib.crc32(name+data)&0xffffffff; return struct.pack('>I',len(data))+name+data+struct.pack('>I',crc)
    raw=b''.join(b'\x00'+bytes(img[y*W*4:(y+1)*W*4]) for y in range(H))
    ihdr=struct.pack('>II',W,H)+bytes([8,6,0,0,0]); compressed=zlib.compress(raw,9)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',ihdr)+chunk(b'IDAT',compressed)+chunk(b'IEND',b'')


_ICON_CACHE = {}

@app.route("/icon.png")
def pwa_icon_192():
    if 192 not in _ICON_CACHE: _ICON_CACHE[192] = _make_mythic_icon_png(192)
    return Response(_ICON_CACHE[192], mimetype="image/png", headers={"Cache-Control": "public, max-age=604800"})

@app.route("/icon-512.png")
def pwa_icon_512():
    if 512 not in _ICON_CACHE: _ICON_CACHE[512] = _make_mythic_icon_png(512)
    return Response(_ICON_CACHE[512], mimetype="image/png", headers={"Cache-Control": "public, max-age=604800"})

@app.route("/favicon.ico")
def favicon():
    if 192 not in _ICON_CACHE: _ICON_CACHE[192] = _make_mythic_icon_png(192)
    return Response(_ICON_CACHE[192], mimetype="image/png", headers={"Cache-Control": "public, max-age=604800"})


@app.route("/api/generate-file", methods=["POST"])
@login_required
def api_generate_file():
    """Generate a downloadable file from text content.
    Supports: pdf, docx/word, txt, md, csv, json, html, js, py, and any other text format.
    Returns base64-encoded file bytes + filename + mimeType."""
    try:
        data = request.get_json(force=True) or {}
        content = (data.get("content") or "").strip()
        fmt = (data.get("format") or "pdf").strip().lower()
        title = (data.get("title") or "Mythic AI Document").strip()[:100]
        if not content:
            return jsonify({"error": "content is required"}), 400

        safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip().replace(" ", "-") or "Mythic-AI"

        # Text / code formats — direct encode, no library needed
        TEXT_FORMATS = {
            "txt": ("text/plain", ".txt"),
            "text": ("text/plain", ".txt"),
            "md": ("text/markdown", ".md"),
            "markdown": ("text/markdown", ".md"),
            "csv": ("text/csv", ".csv"),
            "json": ("application/json", ".json"),
            "html": ("text/html", ".html"),
            "htm": ("text/html", ".html"),
            "css": ("text/css", ".css"),
            "js": ("text/javascript", ".js"),
            "javascript": ("text/javascript", ".js"),
            "py": ("text/x-python", ".py"),
            "python": ("text/x-python", ".py"),
            "ts": ("text/typescript", ".ts"),
            "typescript": ("text/typescript", ".ts"),
            "sql": ("text/plain", ".sql"),
            "yaml": ("text/yaml", ".yaml"),
            "xml": ("text/xml", ".xml"),
            "sh": ("text/x-sh", ".sh"),
            "bash": ("text/x-sh", ".sh"),
            "r": ("text/plain", ".r"),
            "cpp": ("text/x-c++src", ".cpp"),
            "c": ("text/x-csrc", ".c"),
            "java": ("text/x-java", ".java"),
            "rs": ("text/x-rustsrc", ".rs"),
            "rust": ("text/x-rustsrc", ".rs"),
            "php": ("text/x-php", ".php"),
            "rb": ("text/x-ruby", ".rb"),
            "ruby": ("text/x-ruby", ".rb"),
            "go": ("text/x-go", ".go"),
            "swift": ("text/x-swift", ".swift"),
            "kt": ("text/x-kotlin", ".kt"),
            "kotlin": ("text/x-kotlin", ".kt"),
        }
        if fmt in TEXT_FORMATS:
            mime, ext = TEXT_FORMATS[fmt]
            return jsonify({
                "file": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
                "filename": safe_title + ext,
                "mimeType": mime,
            })

        # Word (.docx)
        if fmt in ("docx", "word", "doc"):
            try:
                import docx as _docx, io as _io
                doc = _docx.Document()
                if title: doc.add_heading(title, level=1)
                for para in content.split("\n"): doc.add_paragraph(para)
                buf = _io.BytesIO(); doc.save(buf)
                return jsonify({
                    "file": base64.b64encode(buf.getvalue()).decode("utf-8"),
                    "filename": safe_title + ".docx",
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                })
            except ImportError:
                # python-docx not installed — fall back to .txt
                return jsonify({
                    "file": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
                    "filename": safe_title + ".txt",
                    "mimeType": "text/plain",
                    "note": "Word generation requires `pip install python-docx` on the server. Saved as .txt instead.",
                })

        # PDF — stdlib only (no Pillow/reportlab needed)
        import textwrap as _tw, struct as _struct, zlib as _zlib
        PAGE_W, PAGE_H = 612, 792; MARGIN = 56; FS = 11; LEAD = 15
        max_chars = max(40, int((PAGE_W - 2*MARGIN)/(FS*0.5)))
        max_lines = int((PAGE_H - 2*MARGIN - 40)/LEAD)

        def esc(s): return s.replace("\\","\\\\").replace("(","\\(").replace(")","\\)")
        lines = []
        if title: lines += [("t", title), ("b","")]
        for para in content.split("\n"):
            para = para.rstrip()
            if not para: lines.append(("b",""))
            else: lines += [("x", w) for w in (_tw.wrap(para, max_chars) or [""])]

        pages = [lines[i:i+max_lines] for i in range(0, max(len(lines),1), max_lines)] or [[]]
        objs = []; 
        def emit(d): objs.append(d); return len(objs)
        cat_n = emit(b"ph"); pgs_n = emit(b"ph"); fnt_n = emit(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        pnums = []
        for pg in pages:
            sp = [b"BT", f"/F1 {FS} Tf".encode(), f"{LEAD} TL".encode(), f"{MARGIN} {PAGE_H-MARGIN} Td".encode()]
            first = True
            for kind, text in pg:
                if not first: sp.append(b"T*")
                first = False
                if kind == "b": continue
                if kind == "t": sp.append(b"/F1 16 Tf"); sp.append(f"({esc(text)}) Tj".encode("latin-1","replace")); sp.append(f"/F1 {FS} Tf".encode())
                else: sp.append(f"({esc(text)}) Tj".encode("latin-1","replace"))
            sp.append(b"ET"); stream = b"\n".join(sp)
            cn = emit(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
            pnums.append(emit(f"<< /Type /Page /Parent {pgs_n} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] /Resources << /Font << /F1 {fnt_n} 0 R >> >> /Contents {cn} 0 R >>".encode()))
        kids = " ".join(f"{n} 0 R" for n in pnums)
        objs[pgs_n-1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pnums)} >>".encode()
        objs[cat_n-1] = f"<< /Type /Catalog /Pages {pgs_n} 0 R >>".encode()
        out = bytearray(b"%PDF-1.4\n"); offs = [0]
        for i, od in enumerate(objs, 1): offs.append(len(out)); out += f"{i} 0 obj\n".encode() + od + b"\nendobj\n"
        xr = len(out); out += f"xref\n0 {len(objs)+1}\n".encode() + b"0000000000 65535 f \n"
        for o in offs[1:]: out += f"{o:010d} 00000 n \n".encode()
        out += f"trailer\n<< /Size {len(objs)+1} /Root {cat_n} 0 R >>\nstartxref\n{xr}\n%%EOF".encode()
        return jsonify({
            "file": base64.b64encode(bytes(out)).decode("utf-8"),
            "filename": safe_title + ".pdf",
            "mimeType": "application/pdf",
        })
    except Exception as e:
        return jsonify({"error": f"File generation failed: {e}"}), 500


@app.route("/")
@login_required
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")


@app.route("/api/conversations", methods=["GET"])
@login_required
def api_list_conversations():
    return jsonify({"conversations": list_conversations(current_username())})


@app.route("/api/conversations/<conv_id>", methods=["GET"])
@login_required
def api_get_conversation(conv_id):
    data = load_conversation(current_username(), conv_id)
    if data is None:
        return jsonify({"error": "not found"}), 404
    simplified = []
    for m in data.get("messages", []):
        role = "user" if m["role"] == "user" else "ai"
        text_parts = [p.get("text", "") for p in m["parts"] if "text" in p]
        entry = {"role": role, "text": "".join(text_parts)}
        if m.get("attachment_meta"):
            entry["attachment"] = m["attachment_meta"]
        simplified.append(entry)
    return jsonify({"messages": simplified, "title": data.get("title", "New chat")})


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
@login_required
def api_delete_conversation(conv_id):
    delete_conversation(current_username(), conv_id)
    return jsonify({"status": "deleted"})


@app.route("/api/conversations/<conv_id>", methods=["PATCH"])
@login_required
def api_rename_conversation(conv_id):
    data = request.get_json(force=True) or {}
    new_title = (data.get("title") or "").strip()[:120]
    if not new_title:
        return jsonify({"error": "title is required"}), 400
    username = current_username()
    conv = load_conversation(username, conv_id)
    if conv is None:
        return jsonify({"error": "not found"}), 404
    conv["title"] = new_title
    save_conversation(username, conv_id, conv)
    return jsonify({"status": "renamed", "title": new_title})


def to_ollama_messages(gemini_messages, system_prompt):
    """Convert our stored Gemini-style messages ({role, parts:[...]}) into
    Ollama's chat format ({role, content, images?})."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in gemini_messages:
        role = "user" if m["role"] == "user" else "assistant"
        text = "".join(p.get("text", "") for p in m["parts"] if "text" in p)
        entry = {"role": role, "content": text}
        images = [
            p["inline_data"]["data"]
            for p in m["parts"]
            if "inline_data" in p and p["inline_data"].get("mime_type", "").startswith("image/")
        ]
        if images:
            entry["images"] = images
        msgs.append(entry)
    return msgs


def ollama_stream_chunks(messages):
    """Yields plain text increments from a local Ollama server."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
            stream=True,
            timeout=120,
        )
    except requests.RequestException as e:
        yield (
            f"[Could not reach Ollama at {OLLAMA_URL}: {e}. "
            f"Make sure Ollama is installed and running (`ollama serve`), "
            f"and that you've pulled the model (`ollama pull {OLLAMA_MODEL}`).]"
        )
        return

    if resp.status_code != 200:
        yield f"[Ollama error ({resp.status_code}): {resp.text}]"
        return

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if obj.get("error"):
            yield f"[Ollama error: {obj['error']}]"
            return
        content = obj.get("message", {}).get("content", "")
        if content:
            yield content
        if obj.get("done"):
            break


def to_openai_messages(gemini_messages, system_prompt):
    """Convert stored Gemini-format messages to OpenAI-compatible chat format.
    Used by Groq, OpenRouter, and HuggingFace (all use the same OpenAI-style API)."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in gemini_messages:
        role = "user" if m["role"] == "user" else "assistant"
        text = "".join(p.get("text", "") for p in m["parts"] if "text" in p)
        msgs.append({"role": role, "content": text})
    return msgs


def groq_stream_chunks(messages):
    """Stream from Groq API (OpenAI-compatible, very fast, generous free tier)."""
    if not GROQ_API_KEY:
        return
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    content = obj["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return  # success — stop here
        # rate limited or error — fall through (return without yielding)
    except requests.RequestException:
        pass


def openrouter_stream_chunks(messages):
    """Stream from OpenRouter (aggregates many free models)."""
    if not OPENROUTER_API_KEY:
        return
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
                     "HTTP-Referer": "http://localhost:5000", "X-Title": "Ꮇʏᴛʜɪᴄ ᴀɪ"},
            json={"model": OPENROUTER_MODEL, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    content_chunk = obj["choices"][0]["delta"].get("content", "")
                    if content_chunk:
                        yield content_chunk
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return
        else:
            yield f"[OpenRouter error {resp.status_code}: {resp.text[:200]}]"
            return
    except requests.RequestException as e:
        yield f"[OpenRouter connection error: {e}]"
        return


def huggingface_stream_chunks(messages):
    """Stream from Hugging Face Inference API (free tier available)."""
    if not HF_API_KEY:
        return
    # HF uses the same OpenAI-compatible endpoint format
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"},
            json={"model": HF_MODEL, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    content_chunk = obj["choices"][0]["delta"].get("content", "")
                    if content_chunk:
                        yield content_chunk
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return
        else:
            yield f"[OpenRouter error {resp.status_code}: {resp.text[:200]}]"
            return
    except requests.RequestException as e:
        yield f"[OpenRouter connection error: {e}]"
        return



def cerebras_stream_chunks(messages):
    """Stream from Cerebras AI (very fast, generous free tier, works on servers)."""
    if not CEREBRAS_API_KEY:
        return
    try:
        resp = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
            json={"model": CEREBRAS_MODEL, "messages": messages, "stream": True, "max_tokens": 2048},
            stream=True, timeout=60,
        )
        if resp.status_code == 200:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    chunk = obj["choices"][0]["delta"].get("content", "")
                    if chunk:
                        yield chunk
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return
        else:
            yield f"[Cerebras error {resp.status_code}: {resp.text[:300]}]"
            return
    except requests.RequestException as e:
        yield f"[Cerebras connection error: {e}]"
        return


# --- Round-robin provider rotation ------------------------------------------
# Tracks which provider index to try FIRST next time (persists for the life
# of the process; resets on server restart, which is fine).
_provider_index = [0]


def auto_stream_chunks(gemini_payload, gemini_messages, system_prompt=None, user_api_key=None):
    """Groq first, Cerebras silent fallback. If the user has set their own API key
    in Settings → My API Keys, it is used instead of the server-configured key for
    their chosen provider. Never exposes provider errors to the user."""
    sp = system_prompt or SYSTEM_PROMPT
    openai_msgs = to_openai_messages(gemini_messages, sp)

    # If the user supplied their own key for a provider, try that first
    if user_api_key:
        _uk_provider = user_api_key["provider"]
        _uk_key      = user_api_key["key"]
        _uk_url      = ("https://api.groq.com/openai/v1/chat/completions"
                        if _uk_provider == "groq"
                        else "https://api.cerebras.ai/v1/chat/completions")
        _uk_model    = GROQ_MODEL if _uk_provider == "groq" else CEREBRAS_MODEL
        collected = False
        try:
            resp = requests.post(
                _uk_url,
                headers={"Authorization": f"Bearer {_uk_key}", "Content-Type": "application/json"},
                json={"model": _uk_model, "messages": openai_msgs, "stream": True, "max_tokens": 2048},
                stream=True, timeout=60,
            )
            if resp.status_code == 200:
                for raw_line in resp.iter_lines(decode_unicode=False):
                    if not raw_line: continue
                    try: line = raw_line.decode("utf-8")
                    except: continue
                    if not line.startswith("data:"): continue
                    ds = line[5:].strip()
                    if ds == "[DONE]": break
                    try:
                        obj = json.loads(ds)
                        chunk = obj["choices"][0]["delta"].get("content", "")
                        if chunk: collected = True; yield chunk
                    except: continue
        except Exception:
            pass
        if collected:
            return
        # User key failed — fall through to server keys silently

    order = []
    if PROVIDER in ("auto", "groq") and GROQ_API_KEY:
        order.append(("Groq", lambda: groq_stream_chunks(openai_msgs)))
    if PROVIDER in ("auto", "cerebras") and CEREBRAS_API_KEY:
        order.append(("Cerebras", lambda: cerebras_stream_chunks(openai_msgs)))

    if not order:
        yield "I'm not able to respond right now — no AI provider is configured on the server."
        return

    for _name, fn in order:
        collected = False
        try:
            for chunk in fn():
                collected = True
                yield chunk
            if collected:
                return
        except Exception:
            pass

    yield "I'm having trouble reaching the AI service right now — please try again in a moment."

    if not all_providers:
        yield "[No AI providers configured. Add at least one API key.]"
        return

    n = len(all_providers)
    start = _provider_index[0] % n

    # Try each provider starting from the current rotation position
    for i in range(n):
        idx = (start + i) % n
        name, fn = all_providers[idx]
        collected = []
        try:
            for chunk in fn():
                collected.append(chunk)
                yield chunk
            if collected:
                # Success — next request starts from the NEXT provider (true rotation)
                _provider_index[0] = (idx + 1) % n
                return
        except Exception:
            pass
        # This provider failed — silently try the next one

    yield "[All AI providers failed or are rate-limited. Try again in a moment.]"




def gemini_stream_chunks(payload):
    """Yields plain text increments from Gemini's SSE stream.
    Returns nothing (silently) on auth/quota/rate-limit errors so the
    round-robin rotation automatically falls through to the next provider."""
    try:
        resp = requests.post(
            GEMINI_STREAM_URL,
            params={"key": API_KEY, "alt": "sse"},
            json=payload,
            stream=True,
            timeout=60,
        )
    except requests.RequestException:
        return  # network error — silently fall through

    if resp.status_code != 200:
        # Auth errors (401/403), quota errors (429), and server errors (5xx)
        # all mean "try the next provider" — don't yield anything.
        return

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data_str = raw_line[len("data:"):].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            obj = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        try:
            for part in obj["candidates"][0]["content"]["parts"]:
                if "text" in part:
                    yield part["text"]
        except (KeyError, IndexError):
            continue


# --- Model selector (cosmetic tiers over the same underlying providers) -----
# "VIP" is gated by a password so it isn't just a free option in the dropdown.
# Set VIP_PASSWORD as an environment variable — if it's never set, the VIP
# tier simply can't be unlocked (safe default, no hardcoded password).
VIP_PASSWORD = os.environ.get("VIP_PASSWORD", "")

MODEL_CATALOG = [
    {"id": "mythic-1", "name": "Mythic 1", "vip": False},
    {"id": "mythic-2", "name": "Mythic 2", "vip": False},
    {"id": "mythic-3", "name": "Mythic 3", "vip": False},
    {"id": "mythic-vip", "name": "Mythic VIP 🔒", "vip": True},
]
DEFAULT_MODEL_ID = "mythic-2"


@app.route("/api/models", methods=["GET"])
@login_required
def api_models():
    return jsonify({"models": MODEL_CATALOG, "default": DEFAULT_MODEL_ID})


@app.route("/api/vip-status", methods=["GET"])
@login_required
def api_vip_status():
    return jsonify({"vip": bool(session.get("vip_unlocked"))})


@app.route("/api/vip-unlock", methods=["POST"])
@login_required
def api_vip_unlock():
    data = request.get_json(force=True) or {}
    password = data.get("password") or ""
    if VIP_PASSWORD and password == VIP_PASSWORD:
        session["vip_unlocked"] = True
        session.permanent = True
        return jsonify({"success": True})
    return jsonify({"success": False})


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    conv_id = data.get("conversation_id")
    attachment = data.get("attachment")
    user_name = (data.get("user_name") or "").strip()[:60]
    requested_model = (data.get("model") or DEFAULT_MODEL_ID).strip()
    regenerate = bool(data.get("regenerate"))
    # User's own API key from their browser (set in Settings → My API Keys)
    # Validated strictly: must be a dict with provider + key string, key length ≥ 20.
    user_api_key_raw = data.get("user_api_key")
    user_api_key = None
    if isinstance(user_api_key_raw, dict):
        _provider = (user_api_key_raw.get("provider") or "").strip().lower()
        _key = (user_api_key_raw.get("key") or "").strip()
        if _provider in ("groq", "cerebras") and len(_key) >= 20:
            user_api_key = {"provider": _provider, "key": _key}

    if regenerate:
        if not conv_id:
            return jsonify({"error": "conversation_id is required to regenerate"}), 400
    elif not user_message and not attachment:
        return jsonify({"error": "message or attachment is required"}), 400

    if attachment:
        try:
            raw = base64.b64decode(attachment.get("dataBase64", ""), validate=True)
        except Exception:
            return jsonify({"error": "invalid attachment data"}), 400
        if len(raw) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "attachment too large (max 8MB)"}), 400

    username = current_username()
    conv = load_conversation(username, conv_id) if conv_id else None
    if conv is None:
        if regenerate:
            return jsonify({"error": "conversation not found"}), 404
        conv_id = str(uuid.uuid4())
        conv = {"title": make_title(user_message), "messages": []}

    messages = conv.setdefault("messages", [])

    if regenerate:
        # Drop the most recent assistant reply (if any) so a fresh one replaces it.
        # Leaves the preceding user message in place to regenerate against.
        if messages and messages[-1]["role"] == "model":
            messages.pop()
        if not messages or messages[-1]["role"] != "user":
            return jsonify({"error": "nothing to regenerate"}), 400
    else:
        user_parts = []
        if user_message:
            user_parts.append({"text": user_message})
        attachment_meta = None
        if attachment:
            mime_type = attachment.get("mimeType", "application/octet-stream")
            user_parts.append({
                "inline_data": {"mime_type": mime_type, "data": attachment["dataBase64"]}
            })
            attachment_meta = {"name": attachment.get("name", "file"), "mimeType": mime_type}

        user_entry = {"role": "user", "parts": user_parts}
        if attachment_meta:
            user_entry["attachment_meta"] = attachment_meta
        messages.append(user_entry)

    # Strip attachment_meta (frontend-only field) before sending to the model
    gemini_contents = [
        {"role": m["role"], "parts": m["parts"]} for m in messages
    ]

    effective_system_prompt = SYSTEM_PROMPT
    if user_name:
        effective_system_prompt += (
            f" The user has told you their preferred name is \"{user_name}\". "
            f"Address them as {user_name} naturally where it fits (e.g. greetings, "
            f"acknowledgements) — don't force it into every single reply."
        )
    if requested_model == "mythic-vip" and session.get("vip_unlocked"):
        effective_system_prompt += (
            " The user is on the VIP tier — feel free to go deeper and be more "
            "thorough than usual when it's helpful, without padding for its own sake."
        )
    # Only the real Gemini call gets the "you have search" instruction — fallback
    # providers don't have real search, so giving them that instruction makes them
    # hallucinate fake tool-call JSON into the visible reply.
    gemini_system_prompt = effective_system_prompt + GEMINI_SEARCH_ADDENDUM

    payload = {
        "contents": gemini_contents,
        "systemInstruction": {"parts": [{"text": gemini_system_prompt}]},
        "tools": [{"google_search": {}}],
    }

    def generate():
        full_reply = []
        if PROVIDER == "ollama":
            chunk_source = ollama_stream_chunks(to_ollama_messages(messages, effective_system_prompt))
        else:
            chunk_source = auto_stream_chunks(payload, messages, effective_system_prompt,
                                               user_api_key=user_api_key)

        for chunk in chunk_source:
            full_reply.append(chunk)
            yield chunk
        messages.append({"role": "model", "parts": [{"text": "".join(full_reply)}]})
        save_conversation(username, conv_id, conv)

    resp = Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")
    resp.headers["X-Conversation-Id"] = conv_id
    return resp


@app.route("/api/temp-image/<img_id>", methods=["GET"])
def serve_temp_image(img_id):
    """Serves a temporarily-stashed upload so NanoBanana's servers can fetch it
    by URL for image-to-image editing. See _store_temp_image() above."""
    entry = _TEMP_IMAGES.get(img_id)
    if not entry:
        return jsonify({"error": "not found or expired"}), 404
    return Response(entry["data"], mimetype=entry["mime_type"])


@app.route("/api/generate-image", methods=["POST"])
@login_required
def generate_image():
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "").strip()
    image_b64 = data.get("imageBase64")  # optional — source photo for Ghibli Me / edits
    mime_type = data.get("mimeType", "image/jpeg")
    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    # Preferred path: NanoBanana. Supports real image-to-image editing, so
    # "Ghibli Me" can actually transform the uploaded photo instead of just
    # generating a generic image from text.
    if NANO_BANANA_API_KEY:
        image_urls = None
        if image_b64:
            try:
                raw = base64.b64decode(image_b64, validate=True)
            except Exception:
                return jsonify({"error": "invalid image data"}), 400
            if len(raw) > MAX_UPLOAD_BYTES:
                return jsonify({"error": "image too large (max 8MB)"}), 400
            img_id = _store_temp_image(raw, mime_type)
            image_urls = [f"{request.host_url.rstrip('/')}/api/temp-image/{img_id}"]

        task_id, err = nano_banana_submit(prompt, image_urls=image_urls)
        if err:
            return jsonify({"error": err}), 502
        result_url, err = nano_banana_poll(task_id)
        if err:
            return jsonify({"error": err}), 502
        try:
            img_resp = requests.get(result_url, timeout=30)
            img_resp.raise_for_status()
            return jsonify({"image": base64.b64encode(img_resp.content).decode("utf-8")})
        except requests.RequestException as e:
            return jsonify({"error": f"Could not download result image: {e}"}), 502

    # Fallback: HuggingFace FLUX.1-schnell — text-to-image only, ignores any
    # uploaded photo (it has no image-to-image mode here).
    if not HF_API_KEY:
        return jsonify({"error": "No image generation provider configured"}), 503
    model = "black-forest-labs/FLUX.1-schnell"
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": prompt},
            timeout=60,
        )
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
            img_b64 = base64.b64encode(resp.content).decode("utf-8")
            return jsonify({"image": img_b64})
        else:
            return jsonify({"error": f"Image generation failed ({resp.status_code})"}), 502
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    active = []
    if PROVIDER in ("auto", "gemini") and GEMINI_API_KEY:
        active.append(f"Gemini({GEMINI_MODEL})")
    if PROVIDER in ("auto", "groq") and GROQ_API_KEY:
        active.append(f"Groq({GROQ_MODEL})")
    if PROVIDER in ("auto", "cerebras") and CEREBRAS_API_KEY:
        active.append(f"Cerebras({CEREBRAS_MODEL})")
    if PROVIDER in ("auto", "openrouter") and OPENROUTER_API_KEY:
        active.append(f"OpenRouter({OPENROUTER_MODEL})")
    if PROVIDER in ("auto", "huggingface") and HF_API_KEY:
        active.append(f"HuggingFace({HF_MODEL})")
    if PROVIDER == "ollama":
        active.append(f"Ollama({OLLAMA_MODEL}@{OLLAMA_URL})")
    providers_str = " → ".join(active) if active else "none configured!"
    image_provider = "NanoBanana (image-to-image supported)" if NANO_BANANA_API_KEY else (
        "HuggingFace FLUX (text-to-image only)" if HF_API_KEY else "none configured!"
    )
    print(f"Starting Ꮇʏᴛʜɪᴄ ᴀɪ at http://localhost:5000")
    print(f"Providers (fallback order): {providers_str}")
    print(f"Image generation: {image_provider}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
