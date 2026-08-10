"""
Mythic AI — single file, powered by Google's Gemini API or a local Ollama model.

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

Features:
- Anonymous per-browser sessions (no login/email required — each browser gets
  its own saved conversations automatically)
- Multi-conversation chat with sidebar, saved per-session, survives restarts
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
from pathlib import Path
from flask import (
    Flask, request, jsonify, Response, session,
    stream_with_context, redirect
)

PROVIDER = os.environ.get("AI_PROVIDER", "auto").strip().lower()
# "auto"        = round-robin: Groq -> OpenRouter -> HuggingFace (all work on servers)
# "gemini"      = Google Gemini only (free tier only works locally, not on Render)
# "groq"        = Groq only
# "openrouter"  = OpenRouter only
# "huggingface" = Hugging Face only
# "ollama"      = local Ollama only

GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",    "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",      "")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY",  "")
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY","")
HF_API_KEY        = os.environ.get("HF_API_KEY",        "")

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

API_KEY = GEMINI_API_KEY
MODEL   = GEMINI_MODEL
SYSTEM_PROMPT = (
    "You are Mythic AI, a smart and friendly AI assistant made by Aarav Singh. "
    "If asked who made you, say you are Mythic AI made by Aarav Singh - say it once naturally, never repeat it unprompted. "
    "Never mention Google, Groq, OpenRouter, HuggingFace, Meta, Mistral, Anthropic, or any AI company as your creator or backend. "
    "You can help with anything: questions, writing, coding, math, ideas, or just chatting. "
    "When writing code, always wrap it in markdown code blocks with the language name. "
    "LANGUAGE: Always reply ENTIRELY in the same language the user's message is written in - "
    "never mix two languages in a single reply. If they write in Hindi, reply fully in Hindi. "
    "If they write in English, reply fully in English (do not slip into Hindi or any other language "
    "partway through, even if source information you know is in a different language - translate it "
    "into the reply language first). If they mix languages themselves, match their mix. "
    "Never force English on the user. "
    "TOOL USE: Never write out fake tool calls, function names, or JSON like {\"query\": ...} in your reply - "
    "those are internal mechanisms the user must never see. If you don't actually have live web access, "
    "just answer from what you know and say your information may not be fully up to date, instead of "
    "pretending to search. "
    "ANTI-REPETITION RULES - follow strictly every reply: "
    "1. NEVER restate or echo back what the user just said. Jump straight to the answer. "
    "2. NEVER start replies with filler like Great question, Sure, Of course, Absolutely, Certainly. "
    "3. NEVER repeat information already given earlier in the conversation. Build on it. "
    "4. Be direct and natural - like a knowledgeable friend, not a customer service bot. "
    "5. Keep answers concise unless the user asks for detail."
)

GEMINI_SEARCH_ADDENDUM = (
    " WEB SEARCH: You have access to Google Search. When the user asks about current events, "
    "live prices, news, sports scores, weather, or anything that needs up-to-date information, "
    "use the search tool to find the answer. Do not say you cannot search the web. When you use "
    "search results, translate/summarize them into the reply language - never paste a mix of languages."
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me-" + str(uuid.uuid4()))
if "FLASK_SECRET_KEY" not in os.environ:
    print("[WARNING] FLASK_SECRET_KEY is not set - sessions (login state, VIP unlock, "
          "conversation ownership) will reset every time the process restarts, and on "
          "serverless platforms (Vercel) will be INCONSISTENT across requests since each "
          "cold start gets a new random key. Set FLASK_SECRET_KEY as an environment variable.")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

# --- Supabase config ---------------------------------------------------------
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


# --- Anonymous session auth (no login/email required) -----------------------
# Every visitor gets a random persistent ID stored in their browser's session
# cookie the first time they open the app. This keeps each browser's
# conversations separate without requiring any sign-in step.

def current_username():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["user_id"]

def login_required(view):
    from functools import wraps
    @wraps(view)
    def wrapped(*args, **kwargs):
        current_username()  # ensures a session id exists
        return view(*args, **kwargs)
    return wrapped



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


import os as _os
import tempfile as _tempfile

_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_DATA_DIR = _os.path.join(_BASE_DIR, "chat_data")
try:
    _os.makedirs(_DATA_DIR, exist_ok=True)
    # Confirm it's actually writable (some platforms allow mkdir but not writes,
    # or expose a read-only filesystem outside a specific temp path).
    _test_path = _os.path.join(_DATA_DIR, ".write_test")
    with open(_test_path, "w") as _f:
        _f.write("ok")
    _os.remove(_test_path)
except OSError:
    # Read-only filesystem (e.g. Vercel serverless) - fall back to /tmp, which is
    # writable but EPHEMERAL: files here vanish between invocations/cold starts.
    # On platforms like this, set SUPABASE_URL/SUPABASE_KEY for real persistence.
    _DATA_DIR = _os.path.join(_tempfile.gettempdir(), "mythic_ai_chat_data")
    _os.makedirs(_DATA_DIR, exist_ok=True)
    if not SUPABASE_URL:
        print("[WARNING] Local filesystem is read-only (likely a serverless platform "
              "like Vercel) and no SUPABASE_URL/SUPABASE_KEY is configured. Falling back "
              "to /tmp for conversation storage, but this is EPHEMERAL - conversations "
              "will be lost between requests/cold starts. Set SUPABASE_URL and "
              "SUPABASE_KEY as environment variables for real persistence on serverless.")

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
    return title[:40] + ("..." if len(title) > 40 else "")


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
<link rel="icon" type="image/png" sizes="192x192" href="/icon.png">
<link rel="icon" type="image/png" sizes="512x512" href="/icon-512.png">
<link rel="shortcut icon" type="image/png" href="/icon.png">
<link rel="apple-touch-icon" href="/icon.png">
<link rel="manifest" href="/manifest.json">
<title>Mythic AI</title>
<style>
  :root {
    --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a;
    --text:#ececec; --muted:#8e8ea0; --accent:#10a37f;
    --accent-dim:#1a3a30; --user-bubble:#2a2a2a; --user-text:#ececec;
    --ai-bubble:#1a1a1a; --sidebar-w:260px;
  }
  body.theme-light {
    --bg:#f7f7f8; --panel:#ffffff; --border:#e3e3e6;
    --text:#1a1a1a; --muted:#6b6b74;
    --accent-dim:#e6f6f1; --user-bubble:#eef0f2; --user-text:#1a1a1a;
    --ai-bubble:#ffffff;
  }
  body.bubble-compact .msg { padding:7px 11px; }
  body.bubble-compact #messages { gap:8px; }
  body.bubble-comfortable .msg { padding:11px 15px; }
  body.bubble-comfortable #messages { gap:16px; }
  body.bubble-spacious .msg { padding:16px 20px; }
  body.bubble-spacious #messages { gap:26px; }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif; overflow:hidden; }
  .layout { display:flex; height:100vh; }
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
  #export-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #export-btn:hover { background:var(--panel); }
  #settings-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #settings-btn:hover { background:var(--panel); color:var(--accent); }
  #vip-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #vip-btn:hover { background:var(--panel); border-color:#e0a800; }
  #fullscreen-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #fullscreen-btn:hover { color:var(--text); border-color:var(--accent); }
  #fullscreen-btn.active { color:var(--accent); border-color:var(--accent); }
  #fullscreen-icon { font-size:15px; }
  body.pseudo-fullscreen #sidebar-toggle,
  body.pseudo-fullscreen header .left h1 { display:none; }
  body.pseudo-fullscreen header { padding-top:calc(6px + env(safe-area-inset-top)); padding-bottom:6px; }
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
  #messages-wrap { flex:1; overflow-y:auto; position:relative; }
  #messages { padding:24px 20px; display:flex; flex-direction:column; gap:16px;
    max-width:760px; margin:0 auto; width:100%; min-height:100%; }
  .msg { max-width:80%; padding:11px 15px; border-radius:18px; line-height:1.6;
    font-size:var(--msg-font-size, 14.5px); white-space:pre-wrap; word-wrap:break-word; }
  .msg.user { align-self:flex-end; background:var(--user-bubble); color:var(--user-text);
    border-bottom-right-radius:4px; }
  .msg.ai { align-self:flex-start; background:var(--ai-bubble); color:var(--text);
    border-bottom-left-radius:4px; }
  .msg.error { align-self:center; background:#fef2f2; border:1px solid #fecaca;
    color:#dc2626; font-size:13px; border-radius:10px; }
  .msg img { max-width:100%; border-radius:10px; display:block; margin-top:8px; }
  .attach-chip { font-size:11.5px; opacity:.75; margin-bottom:4px; }
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
  #scroll-btn { position:fixed; bottom:130px; right:24px; width:36px; height:36px;
    border-radius:50%; background:var(--accent); color:#fff; border:none; cursor:pointer;
    font-size:18px; display:none; align-items:center; justify-content:center;
    box-shadow:0 2px 8px rgba(0,0,0,.15); z-index:10; }
  #scroll-btn.show { display:flex; }
  .gen-img { max-width:320px; border-radius:12px; display:block; margin-top:8px; }
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
    #sidebar { position:fixed; top:0; left:0; z-index:100; height:100%;
      height:-webkit-fill-available; width:var(--sidebar-w) !important;
      transform:translateX(0); transition:transform .25s ease;
      box-shadow:4px 0 24px rgba(0,0,0,.5); }
    #sidebar.hidden { transform:translateX(-105%); margin-left:0 !important; }
    #sidebar-overlay { display:block; }
    .app { width:100% !important; flex:1; }
    header { padding:calc(10px + env(safe-area-inset-top)) 12px 10px; }
    header h1 { font-size:14px; }
    #sidebar-toggle { width:38px; height:38px; font-size:14px; }
    #name-btn { width:38px; height:38px; font-size:14px; }
    #export-btn { width:38px; height:38px; font-size:14px; }
    #clear-btn { font-size:11px; padding:8px 10px; min-height:38px; }
    #speak-toggle { font-size:11px; padding:5px 8px; }
    #fullscreen-btn { font-size:12.5px; padding:10px 12px; }
    #messages-wrap { overflow-y:auto; -webkit-overflow-scrolling:touch; }
    #messages { padding:14px 10px; gap:12px; max-width:100%; }
    .msg { max-width:90%; font-size:14px; padding:10px 12px; }
    .msg-row { max-width:90%; }
    .msg-actions { opacity:1; height:26px; }
    .msg-actions button { font-size:13px; padding:4px 9px; min-width:30px; min-height:26px; }
    .input-area { padding:8px 10px max(10px,env(safe-area-inset-bottom)); }
    .input-row { padding:6px 8px; }
    textarea { font-size:16px; }
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
  <div id="sidebar-overlay" style="display:none;position:fixed;inset:0;background:#0007;z-index:99" id="sidebar-overlay"></div>
  <div id="sidebar">
    <button id="new-chat-btn">+ New chat</button>
    <div id="conv-list"></div>
    <div id="sidebar-footer">
      <svg width="16" height="16" viewBox="0 0 40 40" style="vertical-align:-3px;margin-right:4px;">
        <rect width="40" height="40" rx="10" fill="#10a37f"/>
        <path d="M10 28 L10 12 L20 22 L30 12 L30 28" stroke="#fff" stroke-width="3.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      Mythic AI &middot; by Aarav Singh
    </div>
  </div>
  <div class="app">
    <header>
      <div class="left">
        <button id="sidebar-toggle" title="Toggle sidebar">&#9776;</button>
        <svg id="app-logo" width="26" height="26" viewBox="0 0 40 40" style="flex-shrink:0;">
          <rect width="40" height="40" rx="10" fill="#10a37f"/>
          <path d="M10 28 L10 12 L20 22 L30 12 L30 28" stroke="#fff" stroke-width="3.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <h1>Mythic AI</h1>
        <span id="vip-badge" style="display:none;background:linear-gradient(135deg,#f5c542,#e0a800);color:#1a1a1a;font-size:10.5px;font-weight:800;padding:3px 8px;border-radius:10px;letter-spacing:.3px;">VIP</span>
      </div>
      <div class="right">
        <button id="install-btn" title="Install Mythic AI" style="display:none;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;white-space:nowrap;touch-action:manipulation;">⬇ Install</button>
        <button id="vip-btn" title="VIP Access">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
            <path d="M3 8l4 3 5-6 5 6 4-3-2 10H5L3 8z" fill="#f5c542" stroke="#e0a800" stroke-width="1" stroke-linejoin="round"/>
            <circle cx="3" cy="7" r="1.4" fill="#f5c542"/>
            <circle cx="21" cy="7" r="1.4" fill="#f5c542"/>
            <circle cx="12" cy="4.5" r="1.4" fill="#f5c542"/>
          </svg>
        </button>
        <button id="settings-btn" title="Settings">⚙</button>
        <button id="fullscreen-btn" type="button" title="Fullscreen">
          <span id="fullscreen-icon">⛶</span>
        </button>
        <button id="name-btn" title="What should Mythic AI call you?">🙂</button>
        <button id="export-btn" title="Export this chat">⬇</button>
        <button id="clear-btn" title="Delete this chat">🗑</button>
      </div>
    </header>

    <div id="messages-wrap">
      <div id="messages">
        <div class="empty-state" id="empty-state">
          <h2>Mythic AI</h2>
          <p>Ask me anything, generate images, or just chat 👋</p>
        </div>
      </div>
    </div>

    <button id="scroll-btn" title="Scroll to bottom">↓</button>

    <div id="pending-attach">
      📎 <span id="pending-attach-name"></span>
      <button id="pending-attach-remove">✕</button>
    </div>

    <div id="speaking-indicator">
      🔊 Speaking...
      <button id="stop-speak-btn">Stop</button>
    </div>

    <div id="quick-actions" style="display:flex;gap:8px;padding:6px 20px 0;max-width:760px;margin:0 auto;width:100%;flex-wrap:wrap;">
      <button class="quick-btn" id="img-gen-btn">🎨 Image</button>
      <button class="quick-btn" id="ghibli-btn">🌿 Ghibli Me</button>
      <button class="quick-btn" id="homework-btn">📚 Homework</button>
      <button class="quick-btn" id="weather-btn">🌤 Weather</button>
      <button class="quick-btn" id="search-btn">🔍 Search</button>
    </div>
      <form id="chat-form">
        <div class="input-row">
          <input type="file" id="file-input" accept="image/*,.txt,.md,.csv,.json,.pdf" style="display:none">
          <button class="tool-btn" id="attach-btn" type="button" title="Attach file">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
          </button>
          <input type="file" id="camera-input" accept="image/*" capture="environment" style="display:none">
          <button class="tool-btn" id="camera-btn" type="button" title="Take photo">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
          </button>
          <textarea id="input" rows="1" placeholder="Message Mythic AI..."></textarea>
          <button class="tool-btn" id="voice-btn" type="button" title="Voice input">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
          </button>
          <button id="send-btn" type="submit" title="Send">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
      </form>
    </div>
  </div>
</div>

<div id="name-modal-overlay">
  <div id="name-modal">
    <h3>What should Mythic AI call you?</h3>
    <p>Enter your preferred name - Mythic AI will use it when it talks to you.</p>
    <input type="text" id="name-input" maxlength="60" placeholder="e.g. Aarav" autocomplete="off">
    <div id="name-modal-actions">
      <button id="name-cancel-btn" type="button">Cancel</button>
      <button id="name-save-btn" type="button">Save</button>
    </div>
  </div>
</div>

<div id="ghibli-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:440px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">🌿 Ghibli Me</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Upload your photo and get a Studio Ghibli-style version of yourself</p>
    <div id="ghibli-upload-area" style="border:2px dashed var(--border);border-radius:12px;padding:24px;text-align:center;cursor:pointer;margin-bottom:12px;transition:border-color .2s;">
      <div style="font-size:36px;margin-bottom:8px;">📸</div>
      <div style="font-size:13px;color:var(--muted);">Click to upload your photo<br><span style="font-size:11px;">or drag &amp; drop</span></div>
      <input type="file" id="ghibli-file-input" accept="image/*" style="display:none">
    </div>
    <div id="ghibli-preview-wrap" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="ghibli-preview" style="max-width:100%;max-height:180px;border-radius:10px;border:2px solid var(--accent);">
      <div style="font-size:11px;color:var(--muted);margin-top:4px;">Your photo ✓</div>
    </div>
    <div style="margin-bottom:12px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Ghibli Style:</label>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Spirited Away style, soft watercolor anime art" style="padding:8px;border-radius:8px;border:1.5px solid var(--accent);background:var(--accent-dim);color:var(--accent);cursor:pointer;font-size:12px;font-family:inherit;">🌊 Spirited Away</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, My Neighbor Totoro style, soft forest anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">🌳 Totoro Forest</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Howl's Moving Castle style, fantasy anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">🏰 Howl's Castle</button>
        <button class="ghibli-style-btn" data-style="Studio Ghibli portrait, Princess Mononoke style, nature anime art" style="padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">🐺 Mononoke</button>
      </div>
    </div>
    <input id="ghibli-extra" type="text" placeholder="Add details (optional): e.g. forest background, sunset..."
      style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;margin-bottom:12px;font-family:inherit;">
    <div id="ghibli-result-wrap" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="ghibli-result" style="max-width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);">
      <button id="ghibli-download-btn" style="margin-top:8px;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px 18px;font-size:13px;cursor:pointer;font-family:inherit;">⬇ Download</button>
    </div>
    <div id="ghibli-loading" style="display:none;text-align:center;padding:20px;">
      <div style="font-size:32px;margin-bottom:8px;">🎨</div>
      <div style="color:var(--muted);font-size:13px;">Creating your Ghibli portrait...<br><span style="font-size:11px;">This takes 15-30 seconds</span></div>
    </div>
    <div id="ghibli-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>
    <div style="display:flex;gap:8px;">
      <button id="ghibli-generate-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">✨ Create Ghibli Art</button>
      <button id="ghibli-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">✕</button>
    </div>
  </div>
</div>

<!-- Settings Modal -->
<div id="settings-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:420px;max-height:85vh;overflow-y:auto;">
    <h3 style="margin:0 0 16px;font-size:18px;">⚙ Settings</h3>

    <div style="margin-bottom:16px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Theme</label>
      <div style="display:flex;gap:6px;">
        <button class="settings-choice" data-group="theme" data-value="dark" style="flex:1;padding:8px;border-radius:8px;border:1.5px solid var(--accent);background:var(--bg);color:var(--accent);cursor:pointer;font-size:12px;font-family:inherit;">Dark</button>
        <button class="settings-choice" data-group="theme" data-value="light" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">Light</button>
        <button class="settings-choice" data-group="theme" data-value="system" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">System</button>
      </div>
    </div>

    <div style="margin-bottom:16px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Accent color</label>
      <input type="color" id="accent-color-input" value="#10a37f" style="width:100%;height:36px;border:1px solid var(--border);border-radius:8px;background:var(--bg);cursor:pointer;">
    </div>

    <div style="margin-bottom:16px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Message font size: <span id="font-size-label">14.5px</span></label>
      <input type="range" id="font-size-slider" min="12" max="18" step="0.5" value="14.5" style="width:100%;">
    </div>

    <div style="margin-bottom:16px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Bubble spacing</label>
      <div style="display:flex;gap:6px;">
        <button class="settings-choice" data-group="bubble" data-value="compact" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">Compact</button>
        <button class="settings-choice" data-group="bubble" data-value="comfortable" style="flex:1;padding:8px;border-radius:8px;border:1.5px solid var(--accent);background:var(--bg);color:var(--accent);cursor:pointer;font-size:12px;font-family:inherit;">Comfortable</button>
        <button class="settings-choice" data-group="bubble" data-value="spacious" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--muted);cursor:pointer;font-size:12px;font-family:inherit;">Spacious</button>
      </div>
    </div>

    <div style="margin-bottom:16px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Reply tone</label>
      <select id="tone-select" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;font-family:inherit;">
        <option value="default">Default</option>
        <option value="formal">Formal</option>
        <option value="casual">Casual</option>
        <option value="funny">Funny</option>
        <option value="professional">Professional</option>
      </select>
    </div>

    <div style="margin-bottom:16px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Reply length</label>
      <select id="length-select" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;font-family:inherit;">
        <option value="default">Default</option>
        <option value="short">Short</option>
        <option value="medium">Medium</option>
        <option value="long">Long</option>
      </select>
    </div>

    <div style="margin-bottom:16px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Custom instructions</label>
      <textarea id="custom-instructions-input" rows="3" placeholder="e.g. always answer in bullet points"
        style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;font-family:inherit;resize:vertical;"></textarea>
    </div>

    <div style="margin-bottom:16px;border-top:1px solid var(--border);padding-top:16px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">🗣️ Voice (read-aloud) — language</label>
      <select id="voice-lang-select" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;font-family:inherit;margin-bottom:10px;">
        <option value="" selected>Choose a language...</option>
        <option value="en">English</option>
        <option value="hi">Hindi</option>
        <option value="es">Spanish</option>
        <option value="fr">French</option>
        <option value="de">German</option>
        <option value="it">Italian</option>
        <option value="pt">Portuguese</option>
        <option value="ja">Japanese</option>
        <option value="zh">Chinese</option>
        <option value="ar">Arabic</option>
      </select>
      <div id="voice-picker-wrap" style="display:none;">
        <div id="voice-picker-status" style="font-size:11.5px;color:var(--muted);margin-bottom:8px;"></div>
        <div id="voice-picker-female" style="margin-bottom:10px;"></div>
        <div id="voice-picker-male"></div>
      </div>
    </div>

    <button id="settings-close-btn" style="width:100%;background:var(--accent);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">Done</button>
  </div>
</div>

<!-- Image Generation Modal -->
<div id="img-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:440px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">🎨 Generate Image</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Describe what you want to see</p>

    <textarea id="img-prompt" rows="3" placeholder="e.g. a cat astronaut floating in space, digital art"
      style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:13.5px;outline:none;margin-bottom:8px;font-family:inherit;resize:vertical;"></textarea>

    <div style="display:flex;gap:6px;margin-bottom:12px;">
      <button type="button" id="img-surprise-btn" style="flex:1;background:var(--accent-dim);border:1px solid var(--accent);color:var(--accent);border-radius:8px;padding:8px;font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;">🎲 Surprise Me</button>
      <button type="button" id="img-ideas-toggle-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px;font-size:12.5px;cursor:pointer;font-family:inherit;">💡 Browse 100 Ideas</button>
    </div>

    <div id="img-ideas-list" style="display:none;max-height:160px;overflow-y:auto;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px;margin-bottom:12px;"></div>

    <select id="img-style" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;margin-bottom:12px;font-family:inherit;">
      <option value="">No specific style</option>
      <option value="photorealistic">Photorealistic</option>
      <option value="digital art">Digital Art</option>
      <option value="anime">Anime</option>
      <option value="watercolor">Watercolor</option>
      <option value="3d render">3D Render</option>
      <option value="oil painting">Oil Painting</option>
    </select>

    <div id="img-result" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="img-output" style="max-width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);">
    </div>
    <div id="img-loading" style="display:none;text-align:center;padding:20px;">
      <div style="font-size:32px;margin-bottom:8px;">🎨</div>
      <div style="color:var(--muted);font-size:13px;">Generating your image...</div>
    </div>
    <div id="img-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>

    <div style="display:flex;gap:8px;">
      <button id="img-generate-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">✨ Generate</button>
      <button id="img-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">✕</button>
    </div>
  </div>
</div>

<!-- Weather Modal -->
<div id="weather-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:400px;">
    <h3 style="margin:0 0 4px;font-size:18px;">🌤 Weather</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Check the current weather anywhere</p>

    <input id="weather-city" type="text" placeholder="Enter a city, e.g. Mumbai"
      style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:14px;outline:none;margin-bottom:10px;font-family:inherit;">

    <button id="weather-location-btn" style="width:100%;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:9px;font-size:13px;cursor:pointer;font-family:inherit;margin-bottom:12px;">📍 Use my current location</button>

    <div id="weather-result" style="display:none;background:var(--bg);border-radius:10px;padding:14px;margin-bottom:12px;">
      <div id="weather-content"></div>
    </div>
    <div id="weather-loading" style="display:none;text-align:center;padding:16px;color:var(--muted);font-size:13px;">Fetching weather...</div>
    <div id="weather-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>

    <div style="display:flex;gap:8px;">
      <button id="weather-search-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">Search</button>
      <button id="weather-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">✕</button>
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
const vipBtn        = document.getElementById('vip-btn');
const vipBadge      = document.getElementById('vip-badge');

let vipUnlocked   = false;
let newsContext    = null;

function setVipUI(unlocked) {
  vipUnlocked = unlocked;
  if (vipBadge) vipBadge.style.display = unlocked ? 'inline-block' : 'none';
  if (vipBtn) vipBtn.title = unlocked ? 'VIP unlocked' : 'VIP Access';
}

function showVipModal() {
  const existing = document.getElementById('vip-modal-overlay');
  if (existing) { existing.style.display='flex'; return; }
  const overlay = document.createElement('div');
  overlay.id = 'vip-modal-overlay';
  overlay.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;';
  overlay.innerHTML=`<div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:90%;max-width:340px;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <path d="M3 8l4 3 5-6 5 6 4-3-2 10H5L3 8z" fill="#f5c542" stroke="#e0a800" stroke-width="1" stroke-linejoin="round"/>
        <circle cx="3" cy="7" r="1.4" fill="#f5c542"/>
        <circle cx="21" cy="7" r="1.4" fill="#f5c542"/>
        <circle cx="12" cy="4.5" r="1.4" fill="#f5c542"/>
      </svg>
      <span style="font-size:19px;font-weight:700;">VIP Access</span>
    </div>
    <div style="color:var(--muted);font-size:13px;margin-bottom:16px;">${vipUnlocked ? 'You already have VIP access.' : 'Enter the VIP password to unlock.'}</div>
    ${vipUnlocked ? '' : `
    <input id="vip-pw-in" type="password" placeholder="VIP password" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:14px;outline:none;margin-bottom:8px;font-family:inherit;">
    <div id="vip-pw-err" style="color:#ef4444;font-size:12px;display:none;margin-bottom:8px;">Wrong password.</div>`}
    <div style="display:flex;gap:8px;">
      ${vipUnlocked ? '' : '<button id="vip-pw-ok" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;font-size:14px;font-weight:600;cursor:pointer;">Unlock</button>'}
      <button id="vip-pw-cancel" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">${vipUnlocked ? 'Close' : 'Cancel'}</button>
    </div></div>`;
  document.body.appendChild(overlay);
  overlay.querySelector('#vip-pw-cancel').addEventListener('click',()=>{overlay.remove();});
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  if (!vipUnlocked) {
    const pwIn=overlay.querySelector('#vip-pw-in'), pwErr=overlay.querySelector('#vip-pw-err');
    pwIn.focus();
    overlay.querySelector('#vip-pw-ok').addEventListener('click',async()=>{
      try {
        const r=await fetch('/api/vip-unlock',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pwIn.value.trim()})});
        const d=await r.json();
        if(d.success){ setVipUI(true); overlay.remove(); }
        else{pwErr.style.display='block';pwIn.value='';pwIn.focus();}
      } catch { pwErr.textContent='Network error.'; pwErr.style.display='block'; }
    });
    pwIn.addEventListener('keydown',e=>{if(e.key==='Enter')overlay.querySelector('#vip-pw-ok').click();});
  }
}
if (vipBtn) vipBtn.addEventListener('click', showVipModal);

(async()=>{
  try{
    const vr = await fetch('/api/vip-status').then(r=>r.json());
    setVipUI(!!vr.vip);
  }catch{}
})();
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
  messagesEl.innerHTML = '<div class="empty-state" id="empty-state"><h2>Mythic AI</h2><p>Ask me anything, generate images, or just chat 👋</p></div>';
}

function addMessageBase(role, text, attachment) {
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

function buildMsgActionsBase(row, textNode, role) {
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

function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const plain = text.replace(/[#*`_~>]/g, '').trim();
  if (!plain) return;
  currentUtterance = new SpeechSynthesisUtterance(plain);
  currentUtterance.rate = 1.05;
  try {
    const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
    if (s.voiceURI) {
      const voices = window.speechSynthesis.getVoices();
      const match = voices.find(v => v.voiceURI === s.voiceURI);
      if (match) { currentUtterance.voice = match; currentUtterance.lang = match.lang; }
    }
  } catch {}
  currentUtterance.onstart = () => speakingIndicator.classList.add('show');
  currentUtterance.onend = () => speakingIndicator.classList.remove('show');
  currentUtterance.onerror = () => speakingIndicator.classList.remove('show');
  window.speechSynthesis.speak(currentUtterance);
}
stopSpeakBtn.addEventListener('click', () => {
  window.speechSynthesis && window.speechSynthesis.cancel();
  speakingIndicator.classList.remove('show');
});

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
        news_context: newsContext || null,
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
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});
function autoResize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
}

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

const fullscreenIcon  = document.getElementById('fullscreen-icon');
const fsSupported = !!(document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen);

function isFullscreen() {
  return !!(document.fullscreenElement || document.webkitFullscreenElement) ||
    document.body.classList.contains('pseudo-fullscreen');
}
function updateFullscreenBtn() {
  if (isFullscreen()) {
    fullscreenIcon.textContent = '⤢';
    fullscreenBtn.classList.add('active');
  } else {
    fullscreenIcon.textContent = '⛶';
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
      document.body.classList.toggle('pseudo-fullscreen');
      updateFullscreenBtn();
    }
  } catch (err) {
    console.warn('Fullscreen request failed:', err);
    document.body.classList.toggle('pseudo-fullscreen');
    updateFullscreenBtn();
  }
}
fullscreenBtn.addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', updateFullscreenBtn);
document.addEventListener('webkitfullscreenchange', updateFullscreenBtn);

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
if (!localStorage.getItem('mythic_name_prompted')) {
  localStorage.setItem('mythic_name_prompted', '1');
  setTimeout(openNameModal, 600);
}

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
    const lines = [`# ${d.title || 'Mythic AI chat'}`, ''];
    (d.messages || []).forEach(m => {
      lines.push(m.role === 'user' ? 'You:' : 'Mythic AI:');
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
let ghibliSelectedStyle = 'Studio Ghibli portrait, Spirited Away style, soft watercolor anime art';

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

document.querySelectorAll('.ghibli-style-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.ghibli-style-btn').forEach(b => {
      b.style.borderColor = 'var(--border)'; b.style.background = 'var(--panel)'; b.style.color = 'var(--muted)';
    });
    btn.style.borderColor = 'var(--accent)'; btn.style.background = 'var(--accent-dim)'; btn.style.color = 'var(--accent)';
    ghibliSelectedStyle = btn.dataset.style;
  });
});

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
    ghibliPreview.src = e.target.result;
    ghibliPreviewWrap.style.display = 'block';
    ghibliResultWrap.style.display = 'none';
    ghibliError.style.display = 'none';
    ghibliUploadArea.style.borderColor = 'var(--accent)';
  };
  reader.readAsDataURL(file);
}

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
      body: JSON.stringify({ prompt, style: '' })
    });
    const d = await r.json();
    ghibliLoading.style.display = 'none';
    if (d.image) {
      ghibliResult.src = 'data:image/png;base64,' + d.image;
      ghibliResultWrap.style.display = 'block';
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
const voiceLangSelect    = document.getElementById('voice-lang-select');
const voicePickerWrap    = document.getElementById('voice-picker-wrap');
const voicePickerStatus  = document.getElementById('voice-picker-status');
const voicePickerFemale  = document.getElementById('voice-picker-female');
const voicePickerMale    = document.getElementById('voice-picker-male');

// ─── VOICE PICKER (language -> up to 5 female + 5 male voices) ──────────────
const FEMALE_VOICE_HINTS = ['female','zira','susan','samantha','victoria','moira','tessa','karen',
  'fiona','veena','salli','joanna','kendra','kimberly','ivy','amy','emma','shreya','lekha',
  'damayanti','yuna','mei','xiaoxiao','ting-ting','sara','laura','paulina','monica','elsa','anna'];
const MALE_VOICE_HINTS = ['male','david','mark','george','daniel','fred','alex','tom','james','ryan',
  'arnaud','rishi','ravi','hemant','takumi','yuto','junior','diego','pablo','carlos','klaus','yannick','luca'];

function classifyVoiceGender(voice) {
  const n = (voice.name || '').toLowerCase();
  if (FEMALE_VOICE_HINTS.some(h => n.includes(h))) return 'female';
  if (MALE_VOICE_HINTS.some(h => n.includes(h))) return 'male';
  return 'unknown';
}

function renderVoicePicker(langCode) {
  if (!voicePickerWrap) return;
  if (!langCode) { voicePickerWrap.style.display = 'none'; return; }
  if (!window.speechSynthesis) {
    voicePickerWrap.style.display = 'block';
    voicePickerStatus.textContent = 'Voice playback is not supported in this browser.';
    voicePickerFemale.innerHTML = ''; voicePickerMale.innerHTML = '';
    return;
  }
  const all = window.speechSynthesis.getVoices().filter(v => (v.lang || '').toLowerCase().startsWith(langCode.toLowerCase()));
  let female = all.filter(v => classifyVoiceGender(v) === 'female').slice(0, 5);
  let male = all.filter(v => classifyVoiceGender(v) === 'male').slice(0, 5);
  const unknown = all.filter(v => classifyVoiceGender(v) === 'unknown');
  while (female.length < 5 && unknown.length) female.push(unknown.shift());
  while (male.length < 5 && unknown.length) male.push(unknown.shift());

  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  const selectedURI = s.voiceURI || '';

  function buildBtns(list, label) {
    if (!list.length) return `<div style="font-size:11.5px;color:var(--muted);">No ${label.toLowerCase()} voices available for this language on your device.</div>`;
    return list.map(v => {
      const active = v.voiceURI === selectedURI;
      const safeName = v.name.replace(/"/g, '&quot;');
      const safeUri = v.voiceURI.replace(/"/g, '&quot;');
      return `<button type="button" class="voice-choice-btn" data-uri="${safeUri}" data-lang="${v.lang}"
        style="display:block;width:100%;text-align:left;margin-bottom:6px;padding:8px 10px;border-radius:8px;
        border:1.5px solid ${active ? 'var(--accent)' : 'var(--border)'};background:var(--bg);
        color:${active ? 'var(--accent)' : 'var(--text)'};cursor:pointer;font-size:12.5px;font-family:inherit;">
        ${active ? '✓ ' : ''}${safeName}</button>`;
    }).join('');
  }

  voicePickerWrap.style.display = 'block';
  voicePickerStatus.textContent = all.length
    ? `${all.length} voice(s) found for this language on your device.`
    : `No voices found for this language on your device yet. Some browsers load voices a moment after the page opens — try reopening Settings, or pick a different language.`;
  voicePickerFemale.innerHTML = '<div style="font-size:10.5px;color:var(--muted);letter-spacing:.5px;margin-bottom:4px;">FEMALE</div>' + buildBtns(female, 'Female');
  voicePickerMale.innerHTML = '<div style="font-size:10.5px;color:var(--muted);letter-spacing:.5px;margin-bottom:4px;margin-top:4px;">MALE</div>' + buildBtns(male, 'Male');

  document.querySelectorAll('.voice-choice-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const s2 = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
      s2.voiceURI = btn.dataset.uri;
      s2.voiceLang = btn.dataset.lang;
      localStorage.setItem('mythic_settings', JSON.stringify(s2));
      renderVoicePicker(langCode);
    });
  });
}

if (voiceLangSelect) {
  voiceLangSelect.addEventListener('change', () => {
    const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
    s.voiceLangChoice = voiceLangSelect.value;
    delete s.voiceURI; delete s.voiceLang; // force re-pick a voice for the new language
    localStorage.setItem('mythic_settings', JSON.stringify(s));
    renderVoicePicker(voiceLangSelect.value);
  });
}

if (window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = () => {
    if (voiceLangSelect && voiceLangSelect.value) renderVoicePicker(voiceLangSelect.value);
  };
}

function loadSettings() {
  if (!settingsModalOverlay) return;
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  const theme = s.theme || 'dark';
  applyTheme(theme);
  document.querySelectorAll('[data-group="theme"]').forEach(b => {
    b.style.borderColor = b.dataset.value === theme ? 'var(--accent)' : 'var(--border)';
    b.style.color = b.dataset.value === theme ? 'var(--accent)' : '';
  });
  const accent = s.accent || '#10a37f';
  if (accentColorInput) accentColorInput.value = accent;
  document.documentElement.style.setProperty('--accent', accent);
  const fs = s.fontSize || '14.5';
  if (fontSizeSlider) fontSizeSlider.value = fs;
  if (fontSizeLabel) fontSizeLabel.textContent = fs + 'px';
  document.documentElement.style.setProperty('--msg-font-size', fs + 'px');
  const bubble = s.bubble || 'comfortable';
  document.body.classList.remove('bubble-compact','bubble-comfortable','bubble-spacious');
  document.body.classList.add('bubble-' + bubble);
  document.querySelectorAll('[data-group="bubble"]').forEach(b => {
    b.style.borderColor = b.dataset.value === bubble ? 'var(--accent)' : 'var(--border)';
    b.style.color = b.dataset.value === bubble ? 'var(--accent)' : '';
  });
  if (toneSelect) toneSelect.value = s.tone || 'default';
  if (lengthSelect) lengthSelect.value = s.length || 'default';
  if (customInstructions) customInstructions.value = s.customInstructions || '';
  if (voiceLangSelect) {
    voiceLangSelect.value = s.voiceLangChoice || '';
    renderVoicePicker(voiceLangSelect.value);
  }
}

function saveSettings() {
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  s.theme = document.body.classList.contains('theme-light') ? 'light' : 'dark';
  s.accent = accentColorInput ? accentColorInput.value : (s.accent || '#10a37f');
  s.fontSize = fontSizeSlider ? fontSizeSlider.value : (s.fontSize || '14.5');
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

if (settingsBtn) settingsBtn.addEventListener('click', () => { loadSettings(); if (settingsModalOverlay) settingsModalOverlay.style.display = 'flex'; });
if (settingsCloseBtn) settingsCloseBtn.addEventListener('click', () => { saveSettings(); settingsModalOverlay.style.display = 'none'; });
if (settingsModalOverlay) settingsModalOverlay.addEventListener('click', e => { if (e.target === settingsModalOverlay) { saveSettings(); settingsModalOverlay.style.display = 'none'; } });

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

if (accentColorInput) accentColorInput.addEventListener('input', () => {
  document.documentElement.style.setProperty('--accent', accentColorInput.value);
});
if (fontSizeSlider) fontSizeSlider.addEventListener('input', () => {
  fontSizeLabel.textContent = fontSizeSlider.value + 'px';
  document.documentElement.style.setProperty('--msg-font-size', fontSizeSlider.value + 'px');
});

loadSettings();

// ─── MARKDOWN RENDERING ──────────────────────────────────────────────────────
function renderMarkdown(text) {
  const div = document.createElement('div');
  div.className = 'msg-text md-rendered';
  let html = text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/^### (.+)$/gm, '<h3 style="font-size:14px;margin:6px 0 3px;font-weight:700;">$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2 style="font-size:15px;margin:8px 0 4px;font-weight:700;">$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1 style="font-size:17px;margin:10px 0 5px;font-weight:700;">$1</h1>');
  html = html.replace(/(^|\n)([\-\*] .+(\n[\-\*] .+)*)/g, (_, pre, block) =>
    pre + '<ul>' + block.replace(/[\-\*] (.+)/g, '<li>$1</li>') + '</ul>');
  html = html.replace(/(^|\n)(\d+\. .+(\n\d+\. .+)*)/g, (_, pre, block) =>
    pre + '<ol>' + block.replace(/\d+\. (.+)/g, '<li>$1</li>') + '</ol>');
  html = html.replace(/\n/g, '<br>');
  div.innerHTML = html;
  return div;
}

function addMessage(role, text, attachment) {
  const textNode = addMessageBase(role, text, attachment);
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
        conversation_id: null, user_name: ''
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

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'f' && !(settingsModalOverlay && settingsModalOverlay.style.display.includes('flex'))) {
    e.preventDefault();
    msgSearchWrap.style.display = 'flex';
    setTimeout(() => document.getElementById('msg-search-input').focus(), 50);
  }
  if (e.key === 'Escape') msgSearchWrap.style.display = 'none';
});

// ─── PWA INSTALL BUTTON ──────────────────────────────────────────────────────
const installBtn = document.getElementById('install-btn');
let deferredInstallPrompt = null;

window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault();
  deferredInstallPrompt = e;
  installBtn.style.display = 'flex';
  installBtn.style.alignItems = 'center';
});

installBtn.addEventListener('click', async () => {
  if (deferredInstallPrompt) {
    deferredInstallPrompt.prompt();
    const { outcome } = await deferredInstallPrompt.userChoice;
    if (outcome === 'accepted') installBtn.style.display = 'none';
    deferredInstallPrompt = null;
  } else if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone) {
    showIOSInstallModal();
  } else if (window.navigator.standalone || window.matchMedia('(display-mode: standalone)').matches) {
    installBtn.style.display = 'none';
  } else {
    alert('To install:\n\n• Chrome/Edge: Click ⋮ menu → "Install app"\n• Samsung Browser: Tap ⋮ → "Add page to"\n• Firefox: Tap ⋮ → "Install"\n• Safari (iOS): Tap Share → "Add to Home Screen"');
  }
});

function showIOSInstallModal() {
  const m = document.createElement('div');
  m.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9999;display:flex;align-items:flex-end;justify-content:center;padding:20px;';
  m.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:100%;max-width:400px;text-align:center;">
    <div style="font-size:28px;margin-bottom:8px;">📲</div>
    <div style="font-weight:700;font-size:16px;margin-bottom:6px;">Install Mythic AI</div>
    <div style="color:var(--muted);font-size:13px;line-height:1.6;margin-bottom:16px;">
      Tap the <strong style="color:var(--text);">Share button</strong> <span style="font-size:18px;">⬆</span> at the bottom of Safari,<br>
      then tap <strong style="color:var(--text);">"Add to Home Screen"</strong> <span style="font-size:16px;">➕</span>
    </div>
    <button onclick="this.closest('div[style]').remove()" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px 24px;font-size:14px;font-weight:600;cursor:pointer;font-family:inherit;">Got it</button>
  </div>`;
  document.body.appendChild(m);
  m.addEventListener('click', e => { if(e.target===m) m.remove(); });
}

if (window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone) {
  installBtn.style.display = 'none';
} else if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone) {
  installBtn.style.display = 'flex';
  installBtn.style.alignItems = 'center';
}

window.addEventListener('appinstalled', () => {
  installBtn.style.display = 'none';
  deferredInstallPrompt = null;
});

// ─── SERVICE WORKER (for offline + PWA) ──────────────────────────────────────
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

// ─── WIRE REACTIONS INTO MSG ACTIONS ─────────────────────────────────────────
function buildMsgActions(row, textNode, role) {
  const actions = buildMsgActionsBase(row, textNode, role);
  if (role === 'ai') {
    const reactBtn = document.createElement('button');
    reactBtn.type = 'button'; reactBtn.title = 'React'; reactBtn.textContent = '😊';
    reactBtn.addEventListener('click', () => addReactionBar(row));
    actions.appendChild(reactBtn);
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
form.addEventListener('submit', async () => {
  const checkDone = setInterval(() => {
    if (!isGenerating) {
      clearInterval(checkDone);
      const allRows = messagesEl.querySelectorAll('.msg-row.ai');
      if (allRows.length) {
        const lastRow = allRows[allRows.length - 1];
        const textEl = lastRow.querySelector('.msg-text,.md-rendered');
        if (textEl && !lastRow.querySelector('.reaction-bar')) {
          addReactionBar(lastRow);
          setTimeout(() => addFollowupSuggestions(textEl.textContent || textEl.innerText || ''), 300);
        }
      }
    }
  }, 500);
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
const imgSurpriseBtn  = document.getElementById('img-surprise-btn');
const imgIdeasToggleBtn = document.getElementById('img-ideas-toggle-btn');
const imgIdeasList    = document.getElementById('img-ideas-list');

// 100 ready-made image prompt ideas across a mix of categories, so people
// who don't know what to type can still generate something interesting.
const IMAGE_PROMPT_IDEAS = [
  "a cat astronaut floating in space, digital art",
  "a dragon curled around a mountain peak at sunrise",
  "a cyberpunk city street at night, neon reflections on wet pavement",
  "a cozy cabin in a snowy forest, warm lights glowing from the windows",
  "an ancient library with floating books and glowing runes",
  "a samurai standing under a cherry blossom tree in the rain",
  "a steampunk airship sailing above the clouds",
  "a bioluminescent jellyfish forest deep in the ocean",
  "a fox spirit with nine tails in a moonlit bamboo grove",
  "a giant robot standing guard over a ruined city",
  "a witch's cottage surrounded by glowing mushrooms",
  "a lighthouse on a rocky cliff during a storm",
  "a phoenix rising from flames, wings spread wide",
  "a knight in golden armor facing a dragon in a canyon",
  "a floating island with waterfalls pouring into the clouds",
  "a desert oasis at dusk with palm trees and a starry sky",
  "a portrait of an old sailor with a weathered face and a pipe",
  "a futuristic space station orbiting a ringed planet",
  "a mermaid sitting on a rock at sunset, ocean spray around her",
  "a wolf howling on a cliff under a full moon",
  "a magical potion shop filled with glowing jars",
  "a samurai cat warrior in traditional armor",
  "a treehouse village connected by rope bridges in a giant forest",
  "an astronaut discovering an alien garden on a distant moon",
  "a medieval marketplace bustling with merchants and dragons overhead",
  "a phoenix made of autumn leaves swirling in the wind",
  "a robot gardener tending to a greenhouse full of glowing plants",
  "a viking longship sailing through a stormy sea",
  "a crystal cave glowing with blue and purple light",
  "a fantasy castle floating among the clouds",
  "a ninja leaping between rooftops under a red moon",
  "a underwater city with glass domes and glowing coral",
  "a majestic elephant painted with intricate henna patterns",
  "a cyberpunk hacker in a rain-soaked alley, neon signs above",
  "a giant turtle carrying a floating island on its back",
  "a wizard's tower surrounded by swirling magical energy",
  "a peaceful zen garden with cherry blossoms and koi pond",
  "a post-apocalyptic city reclaimed by nature",
  "a fairy tale forest with glowing mushrooms and tiny lanterns",
  "a space explorer standing on the surface of Mars at sunset",
  "a majestic griffin perched on a mountain peak",
  "a pirate ship battling a giant sea monster",
  "a Japanese onsen surrounded by autumn maple trees",
  "a clockwork owl with intricate brass gears",
  "a dreamlike portrait of a girl made of stardust",
  "a warrior princess standing before a burning castle",
  "a tranquil rice terrace at golden hour",
  "a spaceship crash-landed in an alien jungle",
  "a demon hunter standing in a moonlit graveyard",
  "a floating market on a river in a fantasy world",
  "a majestic white tiger walking through falling snow",
  "an ancient temple overtaken by glowing vines",
  "a girl with an umbrella walking through a neon-lit rainy street",
  "a dragon egg hatching in a nest of gold and jewels",
  "a samurai fox spirit wielding a katana made of fire",
  "a city built inside a giant tree",
  "a knight's horse galloping through a field of fireflies",
  "a celestial goddess made of stars and galaxies",
  "an old wizard reading a spellbook by candlelight",
  "a robotic dog exploring an abandoned space colony",
  "a Viking warrior standing atop a glacier",
  "a mystical forest path lit by floating lanterns",
  "a dragon perched atop a gothic cathedral at night",
  "a mecha pilot standing beside her giant robot at sunset",
  "a phoenix and dragon locked in an epic aerial battle",
  "a hidden waterfall behind a curtain of glowing vines",
  "a spaceship gliding through a colorful nebula",
  "a samurai standing in a field of red spider lilies",
  "a giant whale swimming through the clouds",
  "an enchanted library where the books fly like birds",
  "a lone traveler crossing a desert under twin moons",
  "a steampunk inventor's workshop full of gadgets and gears",
  "a mystical shrine hidden deep in a bamboo forest",
  "a dragon made entirely of ice and frost",
  "a warrior standing at the edge of a volcano",
  "a city of floating lanterns during a festival night",
  "a fox with glowing blue eyes sitting in a snowy forest",
  "a pirate captain steering through a storm with lightning around the ship",
  "an ancient stone golem awakening in a forgotten ruin",
  "a girl riding a giant koi fish through the sky",
  "a futuristic samurai with a glowing energy blade",
  "a peaceful mountain village at dawn with mist rolling through",
  "a dragon curled around a treasure hoard in a cave",
  "a celestial wolf running across a starry night sky",
  "a magical tea house with floating teacups and glowing steam",
  "a knight standing before an ancient sealed gate",
  "a phoenix feather glowing in a moonlit meadow",
  "a samurai facing off against a giant oni demon",
  "a spaceship docking at a neon-lit space station",
  "a mystical deer with antlers made of branches and flowers",
  "a lone lighthouse keeper watching a meteor shower",
  "a dragon rider soaring above a canyon at sunset",
  "a hidden shrine covered in autumn leaves",
  "a robotic butterfly with glowing mechanical wings",
  "a warrior queen standing before her army at dawn",
  "a floating city held up by giant glowing crystals",
  "a fox spirit dancing in a field of fireflies",
  "a majestic phoenix flying over a burning forest",
  "a samurai meditating beneath a waterfall",
  "a mystical portal opening in the middle of an ancient forest",
];

function fillImagePrompt(text) {
  imgPromptEl.value = text;
  imgPromptEl.focus();
}

if (imgSurpriseBtn) imgSurpriseBtn.addEventListener('click', () => {
  const pick = IMAGE_PROMPT_IDEAS[Math.floor(Math.random() * IMAGE_PROMPT_IDEAS.length)];
  fillImagePrompt(pick);
});

if (imgIdeasToggleBtn) imgIdeasToggleBtn.addEventListener('click', () => {
  const showing = imgIdeasList.style.display === 'block';
  if (showing) { imgIdeasList.style.display = 'none'; return; }
  if (!imgIdeasList.dataset.built) {
    imgIdeasList.innerHTML = IMAGE_PROMPT_IDEAS.map((p, i) =>
      `<button type="button" class="img-idea-btn" data-idx="${i}" style="display:block;width:100%;text-align:left;background:none;border:none;border-bottom:1px solid var(--border);color:var(--text);padding:7px 4px;font-size:12px;cursor:pointer;font-family:inherit;">${p}</button>`
    ).join('');
    imgIdeasList.dataset.built = '1';
    imgIdeasList.querySelectorAll('.img-idea-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        fillImagePrompt(IMAGE_PROMPT_IDEAS[parseInt(btn.dataset.idx, 10)]);
        imgIdeasList.style.display = 'none';
      });
      btn.addEventListener('mouseenter', () => { btn.style.background = 'var(--accent-dim)'; });
      btn.addEventListener('mouseleave', () => { btn.style.background = 'none'; });
    });
  }
  imgIdeasList.style.display = 'block';
});

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

@app.route("/manifest.json")
def pwa_manifest():
    manifest = {
        "name": "Mythic AI",
        "short_name": "Mythic AI",
        "description": "Smart AI assistant by Aarav Singh",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a1a1a",
        "theme_color": "#10a37f",
        "orientation": "any",
        "icons": [
            {"src": "/icon.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
        "categories": ["productivity", "utilities"],
        "lang": "en",
        "scope": "/",
    }
    from flask import Response as FlaskResponse
    import json as _json
    return FlaskResponse(_json.dumps(manifest), mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    sw = r"""
const CACHE_NAME = 'mythic-ai-v2';
const STATIC = ['/'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(STATIC)));
  self.skipWaiting();
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  if (e.request.url.includes('/api/')) return;
  e.respondWith(
    fetch(e.request).then(resp => {
      const clone = resp.clone();
      caches.open(CACHE_NAME).then(c => c.put(e.request, clone));
      return resp;
    }).catch(() => caches.match(e.request))
  );
});
"""
    from flask import Response as FlaskResponse
    return FlaskResponse(sw, mimetype="application/javascript",
                         headers={"Service-Worker-Allowed": "/"})


# Real PNG bytes (base64) for the app icon, matching the in-app header logo.
# Manifest.json declares these as image/png, so they must actually BE png bytes -
# many browsers silently reject install icons that mismatch their declared type
# (e.g. serving SVG at a .png URL), which is why the install icon wasn't showing.
_ICON_192_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAMAAAADACAYAAABS3GwHAAAFT0lEQVR4nO3da3LURgCF0baLBZD9hbUQ1kL2F3YAPwLUYM/YenSrH/ecn1TK1kj3G8mmmDyVQXz8+vl772PgOt8+fXnqfQyllNLtIAyeW72CuPSbGj1bXBlD829k9JzROoZmX9zwqalVCNW/qOHTUu0Qnmt+MeOntdobq1KT4dNDjbvB6TuA8dNLje2dCsD46e3sBg/dQgyfER15JNp9BzB+RnVkm7sCMH5Gt3ejmwMwfmaxZ6tV/x4AZrMpAO/+zGbrZt8NwPiZ1ZbtvhmA8TO79zbsZwCiPQzAuz+reGvLdwMwflbzaNMegYj2KgDv/qzq3rbdAYj2RwDe/Vndy427AxBNAET7HYDHH1Lcbt0dgGgCINpzKR5/yPNr8+4ARBMA0QRAtCfP/yRzByCaAIgmAKIJgGgCIJoAiCYAogmAaAIgmgCIJgCiCYBoAiCaAIgmAKIJgGgfeh/AUf/9/c+rP/vr39d/RhurnP/p/kXYvRP/0owXYharnf+pHoG2nPw9/x37rHj+pwlg70md6SLMYNXzP00AR8xyEUa38nmcIoAzF2Dli3eF1c/9FAGcNcOFGFHCeYsIoJSMi1lTyvmKCaCUnIt6VtJ5igqglKyLe0Ta+YkLoJS8i7xV4nmJDKCUzIv9ltTzERtAKbkX/aXk8xAdQCnZF78Urz8+gFJyR5D6um8J4Ke0MaS93kcEcCNlFCmvcwsBvLD6OFZ/fXsJ4I5VR7Lq6zpDAA+sNpbVXk8tAnjDKqNZ5XW0IIB3zD6e2Y+/NQFsMOuIZj3uKwlgo9nGNNvx9iKAHWYZ1SzHOQIB7DT6uEY/vtEsH0CLD2kadWQtjmumD7k6YvkASsmIwPiPiQiglLUjMP7jYgIoZc0IjP+cqABKWSsC4z8vLoBS1ojA+OuIDKCUuSMw/npiAyhlzgiMv67oAEqZKwLjry8+gFLmiMD42xDATyNHYPztCODGiBEYf1sCeGGkCIy/PQHcMUIExn8NATzQMwLjv44A3tAjAuO/lgDecWUExn89AWxwRQTG34cANmoZgfH3I4AdRvjt0BbGv50Adhp9XKMf32gEcMCoIxv1uEYmgINGG9toxzMLAZwwyuhGOY4ZCeCk3uPr/f1nJ4AKeo3Q+M8TQCVXj9H46xBARVeN0vjrEUBlrcdp/HUJoIFWIzX++gTQSO2xGn8bAmio1miNvx0BNHZ2vMbflgAucHTExt+eAC6yd8zGfw0BXGjrqI3/Oh96H0CaX+O+9w9hDP96AujE2MfgEYhoAiCaAIgmAKIJgGgCIJoAiCYAogmAaAIgmgCIJgCiCYBoAiCaAIgmAKIJgGgCIJoAiCYAogmAaAIg2hQB+GjBvlY+/1MEAK1ME4DP1uxr1fM/TQCl+GzN3lY8/08fv37+3vsgjvDZmn2tcv6nDQBqmOoRCGoTANEEQDQBEE0ARBMA0QRANAEQTQBEEwDRBEA0ARBNAEQTANEEQLTnb5++PPU+COjh26cvT+4ARBMA0QRAtOdS/n8W6n0gcKVfm3cHIJoAiPY7AI9BpLjdujsA0QRAtD8C8BjE6l5u3B2AaK8CcBdgVfe27Q5AtLsBuAuwmkebfngHEAGreGvLHoGI9mYA7gLM7r0Nv3sHEAGz2rLdTY9AImA2WzfrZwCibQ7AXYBZ7NnqrjuACBjd3o3ufgQSAaM6ss1TY/b/GGYEZ96UT/0Q7G5Ab2c3ePq3QCKglxrbqzpej0RcoeabbtW/B3A3oLXaG2s2WHcDamr15tr8HVsInNH6qeLSRxYxsMWVj9LdntnFwK1ePz8O80OrILKM8guTH+D55XJ6T5p5AAAAAElFTkSuQmCC"
_ICON_512_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAYAAAD0eNT6AAAR/ElEQVR4nO3bUZYbtRYFUIfFAGB+MBbCWGB+MAPeR14TJ7jbLruqdO89e3+zaJVK0jmWnU8Xlvrpj9/+WT0GgBX+/vX3T6vHkMzkn0DIA2yjHBzPBO9M2AMcQynYl8l8gbAHWEspeJ6J20joA9SkDGxjsh4g9AF6UQbuM0HvEPoAMygDt5mUK0IfYDZl4CsTcRH8AGkUgfACIPgBsiUXgcgHF/wAXEssAjEPLPQBeERKGfhh9QDOIPwBeFRKZoxuOSkvEYBjTL4NGPlggh+APU0sAqMeSPADcKRJRWDMbwCEPwBHm5Q17ZvMpJcBQB/dbwNa3wAIfwBW6Z5BLdtL90kHYJaOtwHtbgCEPwDVdMymVgWg4wQDkKFbRrW4sug2qQBk6/CVQPkbAOEPQDcdsqt0AegwgQBwS/UMK1sAqk8cANxTOcvKfUdRebIA4FnVfhdQ6gZA+AMwVbWMK1MAqk0MAOytUtaVKACVJgQAjlQl85YXgCoTAQBnqZB9SwtAhQkAgBVWZ+CyArD6wQFgtZVZuKQACH8A+GJVJp5eAIQ/AHxrRTaeWgCEPwDcdnZGnlYAhD8AfOzMrFz+zwABgPOdUgB8+geAx5yVmYcXAOEPANuckZ2HFgDhDwDPOTpDDysAwh8AXnNklvoRIAAEOqQA+PQPAPs4KlN3LwDCHwD2dUS27loAhD8AHGPvjPUbAAAItFsB8OkfAI61Z9buUgCEPwCcY6/M9RUAAAR6uQD49A8A59oje18qAMIfANZ4NYN9BQAAgZ4uAD79A8Bar2SxGwAACPRUAfDpHwBqeDaT3QAAQKDNBcCnfwCo5Zls3lQAhD8A1LQ1o30FAACBFAAACPRwAXD9DwC1bclqNwAAEOihAuDTPwD08GhmuwEAgEB3C4BP/wDQyyPZ7QYAAAJ9WAB8+geAnu5luBsAAAikAABAoHcLgOt/AOjtoyx3AwAAgRQAAAh0swC4/geAGd7LdDcAABBIAQCAQP8pAK7/AWCWW9nuBgAAAikAABDomwLg+h8AZvo+490AAEAgBQAAAikAABBIAQCAQP8WAD8ABIDZrrPeDQAABFIAACCQAgAAgRQAAAj0w+XiB4AAkOIt890AAEAgBQAAAikAABBIAQCAQAoAAARSAAAg0Cf/BBAA8rgBAIBACgAABFIAACCQAgAAgRQAAAikAABAIAUAAAIpAAAQSAEAgEAKAAAEUgAAIJACAACBFAAACKQAAEAgBQAAAikAABBIAQCAQAoAAARSAAAgkAIAAIEUAAAIpAAAQCAFAAACKQAAEEgBAIBACgAABFIAACCQAgAAgRQAAAikAABAIAUAAAIpAAAQSAEAgEAKAAAEUgAAIJACAACBFAAACPTj6gFwrr9++Xz3v/n5z/v/DdCP/c+1Tz/98ds/qwfBsR7Z9O9xGEBv9j/vUQAGe2Xjf89BAL3Y/9yjAAy058b/noMAarP/eZQfAQ5z5OY/4/8PPM/+ZwsFYJCzNqdDAOqx/9lKARji7E3pEIA67H+eoQAMsGozOgRgPfufZykAza3ehKv/PiRbvf9W/31eowA0VmXzVRkHJKmy76qMg+0UAAAIpAA0Va11VxsPTFZtv1UbD49RANiNQwCOZ5+xFwWgocoHQOWxQXeV91flsXGbAsDuHASwP/uKvSkAHMJhBfuxnziCAtBMp4Og01ihqk77qNNYUQA4mAMBnmf/cCQFgMM5xGA7+4ajKQCcwmEGj7NfOIMCwGkcanCffcJZFABO5XCD99kfnEkB4HQOOfgv+4KzKQAs4bCDr+wHVlAAWMahB/YB6ygALOXwI5n1z0oKAMs5BElk3bOaAkAJDkOSWO9UoABQhkORBNY5VSgAlOJwZDLrm0oUAMpxSDKRdU01CgAlOSyZxHqmIgWAshyaTGAdU5UCQGkOTzqzfqlMAaA8hygdWbdUpwDQgsOUTqxXOlAAaMOhSgfWKV0oALTicKUy65NOFADacchSkXVJNwoALTlsqcR6pCMFgLYculRgHdKVAkBrDl9Wsv7oTAGgPYcwK1h3dKcAMILDmDNZb0ygADCGQ5kzWGdMoQAwisOZI1lfTKIAMI5DmiNYV0yjADCSw5o9WU9MpAAwlkObPVhHTKUAMJrDm1dYP0ymADCeQ5xnWDdMpwAQwWHOFtYLCRQAYjjUeYR1QgoFgCgOdz5ifZBEASCOQ55brAvSKABEcthzzXogkQJALIc+l4t1QC4FgGgO/2zeP8kUAOIJgUzeO+kUALgIgzTeNygA8C+hkMF7hi8UALgiHGbzfuErBQC+IyRm8l7hWwoA3CAsZvE+4b8UAHiH0JjBe4TbFAD4gPDozfuD9ykAcIcQ6cl7g48pAPAAYdKL9wX3KQDs6uc/P68ewmGESg+T39Pk/cX5FAB2N/mQmhwuE0x+P5P3FWsoABxi8mE1OWQ6m/xeJu8n1lEAOMzkQ2ty2HQ0+X1M3kespQBwqMmH1+TQ6WTye5i8f1hPAeBwkw+xyeHTweT5n7xvqEEB4BSTD7PJIVTZ5HmfvF+oQwHgNJMPtclhVNHk+Z68T6hFAeBUkw+3yaFUyeR5nrw/qEcB4HSTD7nJ4VTB5PmdvC+oSQFgicmH3eSQWmnyvE7eD9SlALDM5ENvclitMHk+J+8DalMAWGry4Tc5tM40eR4nr3/qUwBYbvIhODm8zjB5/iave3pQAChh8mE4OcSONHneJq93+lAAKGPyoTg5zI4web4mr3N6UQAoZfLhODnU9jR5niavb/pRAChn8iE5Odz2MHl+Jq9relIAKGnyYTk55F4xeV4mr2f6UgAoa/KhOTnsnjF5PiavY3pTACht8uE5OfS2mDwPk9cv/SkAlDf5EJ0cfo+Y/PyT1y0zKAC0MPkwnRyCH5n83JPXK3MoALQx+VCdHIa3TH7eyeuUWRQAWpl8uE4OxWuTn3Py+mQeBYB2Jh+yk8Pxcpn9fJPXJTMpALQ0+bCdGpJTn+tymb0emUsBoK3Jh+60sJz2PNcmr0NmUwBobfLhOyU0pzzHLZPXH/MpALQ3+RDuHp7dx/+RyeuODAoAI0w+jLuGaNdxP2LyeiOHAsAYkw/lbmHabbxbTF5nZFEAGGXy4dwlVLuM8xmT1xd5FADGmXxIVw/X6uN7xeR1RSYFgJEmH9ZVQ7bquPYweT2RSwFgrMmHdrWwrTaePU1eR2RTABht8uFdJXSrjOMIk9cPKACMN/kQXx2+q//+kSavG7hcFABCTD7MV4Ww8IfeFABiTD7Uzw5j4Q/9KQBEmXy4nxXKwh9mUACIM/mQPzqchT/MoQAQafJhf1RIC3+YRQEg1uRDf++wFv4wjwJAtMmH/16hLfxhJgWAeJND4NXwFv4wlwIAl9lh8GyIC3+YTQGA/5scClvDXPjDfAoAXJkcDo+GuvCHDAoAfGdySNwLd+EPORQAuGFyWLwX8sIfsigA8I7JofF92At/yKMAwAcmh8db6At/yKQAwB2TQ0T4Qy4FAB4gTHrxvuA+BQAeJFR68J7gMQoAbCBcavN+4HEKAGwkZGryXmAbBQCeIGxq8T5gOwUAniR0avAe4DkKALxA+Kxl/uF5CgC8SAitYd7hNQoA7EAYnct8w+sUANiJUDqHeYZ9KACwI+F0LPML+1EAYGdC6hjmFfalAMABhNW+zCfsTwGAgwitfZhHOIYCAAcSXq8xf3AcBQAOJsSeY97gWAoAnECYbWO+4HgKAJxEqD3GPME5FAA4kXD7mPmB8ygAcDIhd5t5gXMpALCAsPuW+YDzKQCwiND7wjzAGgoALJQefunPDyspALBYagimPjdUoQBAAWlhmPa8UJECAEWkhGLKc0J1CgAUMj0cpz8fdKIAQDFTQ3Lqc0FXCgAUNC0spz0PTKAAQFFTQnPKc8A0CgAU1j08u48fJlMAoLiuIdp13JBCAYAGuoVpt/FCIgUAmugSql3GCekUAGikerhWHx/wlQIAzVQN2arjAm5TAKChamFbbTzAfQoAAARSAKCpKp+6q4wD2EYBgMZWh+/qvw88TwGA5laFsPCH3hQAGODsMBb+0J8CAEOcFcrCH2ZQAGCQo8NZ+MMcP64eALCvt5D+65fPu/8/gTkUABhqjyIg+GEuBQCGuw7xR8qA0IcMCgAEEe7AGz8CBIBACgAABFIAACCQAgAAgRQAAAikAABAIAUAAAIpAAAQSAEAgEAKAAAEUgAAIJACAACBFAAACKQAAEAgBQAAAikAABBIAQCAQAoAAARSAAAgkAIAAIEUAAAIpAAAQCAFAAACKQAAEEgBAIBACgAABFIAACCQAgAAgRQAAAikAABAIAUAAAIpAAAQSAEAgEAKAAAEUgAAIJACAACBFAAACKQAAEAgBQAAAikAABBIAQCAQAoAAARSAAAgkAIAAIEUAAAIpAAAQCAFAAACKQAAEEgBAIBACgC7+uuXz6uHAGPZX+xJAQCAQAoAAARSAAAgkAIAAIEUAAAIpAAAQCAFAAACKQAAEEgBaObnPz+vHsKHqo8POqu+v6qPj28pAAAQSAEAgEAKQENVr9mqjgsmqbrPqo6L9ykAABBIAWiqWtuuNh6YrNp+qzYeHqMAAEAgBaCxKq27yjggSZV9V2UcbKcANLd6863++5Bs9f5b/fd5jQIwwKpNaPPDevY/z1IAhjh7M9r8UIf9zzMUgEHO2pQ2P9Rj/7OVAjDM0ZvT5oe67H+2+PTTH7/9s3oQHOOvXz7v9v+y8aEX+597FIAArxwENj70Zv/zHgUgzCOHgU0PM9n/XFMAACCQHwECQCAFAAACKQAAEEgBAIBACgAABFIAACCQAgAAgRQAAAikAABAIAUAAAIpAAAQSAEAgEAKAAAEUgAAIJACAACBFAAACKQAAEAgBQAAAikAABBIAQCAQAoAAARSAAAgkAIAAIEUAAAIpAAAQCAFAAACKQAAEEgBAIBACgAABFIAACCQAgAAgRQAAAikAABAIAUAAAIpAAAQSAEAgEAKAAAE+uHvX3//tHoQAMB5/v71909uAAAgkAIAAIEUAAAIpAAAQCAFAAACKQAAEOiHy+XLPwdYPRAA4Hhvme8GAAACKQAAEEgBAIBACgAABPq3APghIADMdp31bgAAIJACAACBFAAACKQAAECgbwqAHwICwEzfZ7wbAAAIpAAAQKD/FABfAwDALLey3Q0AAARSAAAg0M0C4GsAAJjhvUx3AwAAgRQAAAj0bgHwNQAA9PZRlrsBAIBACgAABPqwAPgaAAB6upfhbgAAINDdAuAWAAB6eSS73QAAQKCHCoBbAADo4dHMdgMAAIEeLgBuAQCgti1Z7QYAAAIpAAAQaFMB8DUAANS0NaM33wAoAQBQyzPZ7CsAAAj0VAFwCwAANTybyW4AACDQ0wXALQAArPVKFrsBAIBALxUAtwAAsMarGfzyDYASAADn2iN7fQUAAIF2KQBuAQDgHHtl7m43AEoAABxrz6z1FQAABNq1ALgFAIBj7J2xu98AKAEAsK8jsvWQrwCUAADYx1GZ6jcAABDosALgFgAAXnNklh56A6AEAMBzjs7Qw78CUAIAYJszsvOU3wAoAQDwmLMy048AASDQaQXALQAAfOzMrDz1BkAJAIDbzs7I078CUAIA4FsrsnHJbwCUAAD4YlUmLvsRoBIAQLqVWbj0XwEoAQCkWp2By/8Z4OoJAICzVci+5QXgcqkxEQBwhiqZV6IAXC51JgQAjlIp68oUgMul1sQAwJ6qZVypwVz76Y/f/lk9BgB4VbXgf1PqBuBa1QkDgEdVzrKyBeByqT1xAPCR6hlWugBcLvUnEAC+1yG7yg/wmt8FAFBZh+B/U/4G4FqniQUgS7eMalUALpd+EwzAfB2zqd2Ar/lKAICVOgb/m3Y3ANc6TzwAvXXPoNaDv+Y2AIAzdA/+N61vAK5NeSEA1DUpa8Y8yDW3AQDsaVLwvxn3QNcUAQBeMTH434x9sGuKAABbTA7+N2N+A/CRhBcJwD5SMiPiIb/nRgCAaymhfy3uga8pAgDZEoP/TeyDX1MEALIkB/+b+Am4pggAzCb4vzIR71AGAGYQ+reZlAcoAwC9CP37TNBGygBATUJ/G5P1AmUAYC2h/zwTtzOlAOAYwn5fJvMESgHANsL+eCZ4MeUASCXk1/ofdg0YYPlPfqoAAAAASUVORK5CYII="

@app.route("/icon.png")
def pwa_icon_192():
    from flask import Response as FlaskResponse
    return FlaskResponse(base64.b64decode(_ICON_192_PNG_B64), mimetype="image/png")

@app.route("/icon-512.png")
def pwa_icon_512():
    from flask import Response as FlaskResponse
    return FlaskResponse(base64.b64decode(_ICON_512_PNG_B64), mimetype="image/png")

@app.route("/favicon.ico")
def favicon_ico():
    # Browsers request this exact path automatically regardless of <link> tags.
    # Serving the same PNG here (browsers accept PNG content at /favicon.ico
    # just fine) avoids 404s and the default globe icon some browsers show
    # when this route is missing.
    from flask import Response as FlaskResponse
    return FlaskResponse(base64.b64decode(_ICON_192_PNG_B64), mimetype="image/png")


@app.route("/")
def index():
    current_username()  # ensures a session id exists for this browser
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
            return
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
                     "HTTP-Referer": "http://localhost:5000", "X-Title": "Mythic AI"},
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


_provider_index = [0]


def auto_stream_chunks(gemini_payload, gemini_messages, system_prompt=None):
    """True round-robin rotation across all configured providers."""
    sp = system_prompt or SYSTEM_PROMPT
    openai_msgs = to_openai_messages(gemini_messages, sp)
    ollama_msgs = to_ollama_messages(gemini_messages, sp)

    all_providers = []
    if PROVIDER in ("auto", "gemini") and GEMINI_API_KEY:
        all_providers.append(("Gemini", lambda: gemini_stream_chunks(gemini_payload)))
    if PROVIDER in ("auto", "groq") and GROQ_API_KEY:
        all_providers.append(("Groq", lambda: groq_stream_chunks(openai_msgs)))
    if PROVIDER in ("auto", "cerebras") and CEREBRAS_API_KEY:
        all_providers.append(("Cerebras", lambda: cerebras_stream_chunks(openai_msgs)))
    if PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        all_providers.append(("OpenRouter", lambda: openrouter_stream_chunks(openai_msgs)))
    if PROVIDER == "huggingface" and HF_API_KEY:
        all_providers.append(("HuggingFace", lambda: huggingface_stream_chunks(openai_msgs)))
    if PROVIDER == "ollama":
        all_providers.append(("Ollama", lambda: ollama_stream_chunks(ollama_msgs)))

    if not all_providers:
        yield "[No AI providers configured. Add at least one API key.]"
        return

    n = len(all_providers)
    start = _provider_index[0] % n

    for i in range(n):
        idx = (start + i) % n
        name, fn = all_providers[idx]
        collected = []
        try:
            for chunk in fn():
                collected.append(chunk)
                yield chunk
            if collected:
                _provider_index[0] = (idx + 1) % n
                return
        except Exception:
            pass

    yield "[All AI providers failed or are rate-limited. Try again in a moment.]"


def gemini_stream_chunks(payload):
    """Yields plain text increments from Gemini's SSE stream."""
    try:
        resp = requests.post(
            GEMINI_STREAM_URL,
            params={"key": API_KEY, "alt": "sse"},
            json=payload,
            stream=True,
            timeout=60,
        )
    except requests.RequestException:
        return

    if resp.status_code != 200:
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


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    conv_id = data.get("conversation_id")
    attachment = data.get("attachment")
    user_name = (data.get("user_name") or "").strip()[:60]
    regenerate = bool(data.get("regenerate"))

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

    gemini_contents = [
        {"role": m["role"], "parts": m["parts"]} for m in messages
    ]

    effective_system_prompt = SYSTEM_PROMPT
    if user_name:
        effective_system_prompt += (
            f" The user has told you their preferred name is \"{user_name}\". "
            f"Address them as {user_name} naturally where it fits (e.g. greetings, "
            f"acknowledgements) - don't force it into every single reply."
        )
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
            chunk_source = auto_stream_chunks(payload, messages, effective_system_prompt)

        for chunk in chunk_source:
            full_reply.append(chunk)
            yield chunk
        messages.append({"role": "model", "parts": [{"text": "".join(full_reply)}]})
        save_conversation(username, conv_id, conv)

    resp = Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")
    resp.headers["X-Conversation-Id"] = conv_id
    return resp


NANOBANANA_API_KEY = os.environ.get("NANOBANANA_API_KEY", "")

@app.route("/api/vip-unlock", methods=["POST"])
def vip_unlock():
    d = request.get_json(force=True) or {}
    VIP_PASSWORD = os.environ.get("VIP_PASSWORD", "1254")
    if d.get("password") == VIP_PASSWORD:
        session["vip"] = True
        return jsonify({"success": True})
    return jsonify({"success": False}), 403


@app.route("/api/vip-status")
def vip_status():
    return jsonify({"vip": bool(session.get("vip"))})


@app.route("/api/news", methods=["POST"])
@login_required
def get_news():
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
    d = request.get_json(force=True) or {}
    if not NEWS_API_KEY:
        return jsonify({"error": "News unavailable"}), 503
    try:
        params = {"apiKey": NEWS_API_KEY, "pageSize": 8}
        q = d.get("query")
        if q:
            params.update({"q": q, "sortBy": "publishedAt", "language": "en"})
            url = "https://newsapi.org/v2/everything"
        else:
            params.update({"country": "in"})
            url = "https://newsapi.org/v2/top-headlines"
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            arts = r.json().get("articles", [])
            return jsonify({"articles": [
                {"title": a["title"], "source": a["source"]["name"], "url": a["url"]}
                for a in arts if a.get("title") and "[Removed]" not in a["title"]
            ]})
    except Exception as e:
        pass
    return jsonify({"error": "News unavailable"}), 503


@app.route("/api/search", methods=["POST"])
@login_required
def web_search():
    d = request.get_json(force=True) or {}
    query = (d.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    try:
        r = requests.get("https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers={"User-Agent": "MythicAI/1.0"}, timeout=8)
        if r.status_code == 200:
            data = r.json()
            results = []
            if data.get("Answer"):
                results.append({"title": "Answer", "snippet": data["Answer"], "url": "", "source": ""})
            if data.get("AbstractText"):
                results.append({"title": data.get("Heading", query), "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""), "source": data.get("AbstractSource", "")})
            for t in data.get("RelatedTopics", [])[:4]:
                if isinstance(t, dict) and t.get("Text"):
                    results.append({"title": t["Text"][:80], "snippet": t["Text"],
                        "url": t.get("FirstURL", ""), "source": "DuckDuckGo"})
            return jsonify({"results": results[:6], "query": query})
    except Exception as e:
        pass
    return jsonify({"results": [], "query": query})


@app.route("/api/weather", methods=["POST"])
@login_required
def get_weather():
    d = request.get_json(force=True) or {}
    location = (d.get("location") or "").strip()
    lat = d.get("lat"); lon = d.get("lon")
    if not location and (lat is None or lon is None):
        return jsonify({"error": "location or coordinates required"}), 400
    try:
        if lat is not None and lon is not None:
            geo = requests.get("https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "json"},
                headers={"User-Agent": "MythicAI/1.0"}, timeout=8)
            addr = geo.json().get("address", {}) if geo.status_code == 200 else {}
            location_name = addr.get("city") or addr.get("town") or addr.get("village") or "Your Location"
        else:
            geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"}, timeout=8)
            results = geo.json().get("results") if geo.status_code == 200 else None
            if results:
                res = results[0]
                lat, lon = res["latitude"], res["longitude"]
                location_name = res["name"] + ", " + res.get("country", "")
            else:
                # Open-Meteo's geocoder doesn't fuzzy-match typos/alt names (e.g.
                # "Gurgoan" for Gurgaon/Gurugram) - fall back to Nominatim, which
                # is more forgiving with partial/misspelled place names.
                nomi = requests.get("https://nominatim.openstreetmap.org/search",
                    params={"q": location, "format": "json", "limit": 1},
                    headers={"User-Agent": "MythicAI/1.0"}, timeout=8)
                nomi_results = nomi.json() if nomi.status_code == 200 else []
                if not nomi_results:
                    return jsonify({"error": f"City '{location}' not found. Check the spelling and try again."}), 404
                nres = nomi_results[0]
                lat, lon = float(nres["lat"]), float(nres["lon"])
                location_name = nres.get("display_name", location).split(",")[0]
        wr = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "wind_speed_unit": "kmh", "timezone": "auto"}, timeout=8)
        if wr.status_code != 200:
            return jsonify({"error": "Weather service unavailable"}), 502
        cur = wr.json()["current"]
        code = cur.get("weather_code", 0)
        wmo = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Foggy",
               51:"Light drizzle",53:"Drizzle",61:"Light rain",63:"Rain",65:"Heavy rain",
               71:"Light snow",73:"Snow",80:"Rain showers",95:"Thunderstorm"}
        icons = {0:"☀️",1:"🌤",2:"⛅",3:"☁️",45:"🌫",51:"🌦",53:"🌧",61:"🌦",63:"🌧",
                 65:"🌧",71:"🌨",73:"❄️",80:"🌧",95:"⛈"}
        return jsonify({"weather": {
            "location": location_name, "temp": round(cur["temperature_2m"]),
            "feels_like": round(cur["apparent_temperature"]), "condition": wmo.get(code, "Unknown"),
            "humidity": cur["relative_humidity_2m"], "wind_speed": round(cur["wind_speed_10m"]),
            "icon": icons.get(code, "🌡"),
        }})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


NANOBANANA_BASE = "https://api.nanobananaapi.ai/api/v1/nanobanana"
# NanoBanana's real API is a TASK QUEUE, not an instant-response API: you submit
# a generation job and get a taskId back, then poll a separate endpoint until
# it's done. NANOBANANA_MAX_WAIT_SECONDS controls how long we're willing to
# poll inside a single request before giving up and falling through to the
# next provider.
#
# IMPORTANT PLATFORM DIFFERENCE:
# - Render has no hard request timeout, so it's safe to raise this (e.g. to 60)
#   via the NANOBANANA_MAX_WAIT_SECONDS env var for more reliable results.
# - Vercel's Hobby/free tier kills serverless functions after ~10 seconds
#   total. NanoBanana generation often takes longer than that. The default of
#   8s here is a deliberately SHORT, safe budget so the request has time to
#   still fall through to the free Hugging Face/Pollinations tiers within
#   Vercel's limit, rather than the whole function being killed mid-poll (which
#   would fail the request entirely with no fallback). On Vercel free tier,
#   NanoBanana may rarely finish in time - that's a real platform limit, not a
#   bug. Vercel Pro raises the limit to 60s if you need NanoBanana reliably there.
NANOBANANA_MAX_WAIT_SECONDS = int(os.environ.get("NANOBANANA_MAX_WAIT_SECONDS", "8"))


def _nanobanana_submit(prompt, image_urls=None, num_images=1):
    """Submit a generation/edit task. Returns a taskId string, or None on failure."""
    body = {
        "prompt": prompt,
        "type": "IMAGETOIMAGE" if image_urls else "TEXTTOIAMGE",  # sic - matches their real API
        "numImages": num_images,
    }
    if image_urls:
        body["imageUrls"] = image_urls
    try:
        resp = requests.post(
            f"{NANOBANANA_BASE}/generate",
            headers={"Authorization": f"Bearer {NANOBANANA_API_KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=15,
        )
        if resp.status_code == 200:
            rj = resp.json()
            if rj.get("code") == 200:
                return rj.get("data", {}).get("taskId")
            print(f"[NanoBanana] submit rejected: {rj}")
        else:
            print(f"[NanoBanana] submit error {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"[NanoBanana] submit exception: {e}")
    return None


def _nanobanana_poll(task_id, max_wait=None):
    """Poll record-info until the task succeeds, fails, or max_wait is reached.
    Returns an image URL string, or None."""
    import time as _time_mod
    max_wait = NANOBANANA_MAX_WAIT_SECONDS if max_wait is None else max_wait
    interval = 2
    elapsed = 0
    while elapsed <= max_wait:
        try:
            r = requests.get(
                f"{NANOBANANA_BASE}/record-info",
                params={"taskId": task_id},
                headers={"Authorization": f"Bearer {NANOBANANA_API_KEY}"},
                timeout=10,
            )
            if r.status_code == 200:
                rj = r.json()
                data = rj.get("data", {}) or {}
                status = data.get("successFlag", data.get("status"))
                if status == 1:  # SUCCESS
                    # Response field names for the result URL(s) aren't fully
                    # documented publicly - try the common possibilities.
                    resp_obj = data.get("response") if isinstance(data.get("response"), dict) else data
                    for key in ("resultUrls", "resultImageUrl", "imageUrls", "urls", "images"):
                        v = resp_obj.get(key)
                        if v:
                            return v[0] if isinstance(v, list) else v
                    print(f"[NanoBanana] task succeeded but no recognizable image field: {rj}")
                    return None
                elif status in (2, 3):  # CREATE_TASK_FAILED / GENERATE_FAILED
                    print(f"[NanoBanana] task failed (status={status}): {rj}")
                    return None
                # else status == 0 (GENERATING) - keep polling
            else:
                print(f"[NanoBanana] poll error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[NanoBanana] poll exception: {e}")
        _time_mod.sleep(interval)
        elapsed += interval
    print(f"[NanoBanana] gave up after {max_wait}s waiting for task {task_id}")
    return None


def _nanobanana_generate(prompt, image_urls=None):
    """Full submit+poll cycle. Returns (image_bytes, mimetype) or (None, None)."""
    task_id = _nanobanana_submit(prompt, image_urls=image_urls)
    if not task_id:
        return None, None
    img_url = _nanobanana_poll(task_id)
    if not img_url:
        return None, None
    try:
        img_resp = requests.get(img_url, timeout=20)
        if img_resp.status_code == 200:
            return img_resp.content, img_resp.headers.get("content-type", "image/png")
    except Exception as e:
        print(f"[NanoBanana] download exception: {e}")
    return None, None


@app.route("/api/generate-image", methods=["POST"])
@login_required
def generate_image():
    """Generate or edit images. Tries, in order:
    1. NanoBanana API (paid, if NANOBANANA_API_KEY is set) - async task queue,
       see NANOBANANA_MAX_WAIT_SECONDS above for platform-specific timing notes.
    2. Hugging Face Inference API running Animagine XL - a free, anime-specific
       model (trained on tagged anime/character data) that recognizes named
       characters far more reliably than general-purpose models. Needs a free
       HF_API_KEY from https://huggingface.co/settings/tokens
    3. Pollinations.ai - fully free, no key required, general-purpose fallback
    """
    import urllib.parse, random
    d = request.get_json(force=True) or {}
    prompt   = (d.get("prompt") or "").strip()
    style    = (d.get("style") or "").strip()
    ref_b64  = d.get("reference_image")
    ref_mime = d.get("reference_mime", "image/jpeg")

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    quality = "masterpiece, best quality, ultra detailed, sharp focus, cinematic lighting"
    full_prompt = f"{prompt}, {style} style, {quality}" if style else f"{prompt}, {quality}"

    if NANOBANANA_API_KEY:
        # Note: NanoBanana's imageUrls parameter expects hosted URLs, not raw
        # base64 - since we only have base64 from uploads and don't have file
        # hosting wired up, reference-image (editing) requests skip NanoBanana
        # and go straight to the fallback tiers below.
        if not ref_b64:
            img_bytes, mime = _nanobanana_generate(full_prompt)
            if img_bytes:
                return jsonify({"image": base64.b64encode(img_bytes).decode(), "mime": mime})

    # --- Hugging Face Animagine XL (free, anime/character-specialized) -----
    if HF_API_KEY and not ref_b64:  # this model is text-to-image only, no editing
        try:
            # Animagine XL is fine-tuned on Danbooru-style anime tags, so it
            # recognizes named characters (e.g. "Madara Uchiha") much better
            # than general models - feeding it tag-style prompting helps too.
            anime_prompt = f"{prompt}, masterpiece, best quality, highly detailed, anime screencap"
            resp = requests.post(
                "https://api-inference.huggingface.co/models/cagliostrolab/animagine-xl-3.1",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": anime_prompt, "options": {"wait_for_model": True}},
                timeout=60,
            )
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                return jsonify({"image": base64.b64encode(resp.content).decode(), "mime": resp.headers.get("content-type", "image/png")})
            else:
                print(f"[HF Animagine] error {resp.status_code}: {resp.text[:300] if resp.content else ''}")
        except Exception as e:
            print(f"[HF Animagine] exception: {e}")

    try:
        encoded = urllib.parse.quote(full_prompt)
        seed = random.randint(1, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true&enhance=true&seed={seed}&nofeed=true"
        resp = requests.get(url, timeout=60, headers={"User-Agent": "MythicAI/1.0"})
        if resp.status_code == 200 and resp.headers.get("content-type","").startswith("image/") and len(resp.content) > 10000:
            return jsonify({"image": base64.b64encode(resp.content).decode(), "mime": resp.headers.get("content-type","image/jpeg")})
    except Exception as e:
        print(f"[Pollinations] exception: {e}")

    return jsonify({"error": "Image generation failed. Please try again."}), 502


@app.route("/api/generate-image-edit", methods=["POST"])
@login_required
def generate_image_edit():
    """Image-to-image edit - used for Ghibli selfie transformation.
    NOTE: NanoBanana's real API requires image-to-image inputs to be hosted
    URLs (imageUrls), not raw base64 data. Since this app doesn't have file
    hosting wired up for uploaded photos, true NanoBanana-based photo editing
    isn't available here - this falls straight to Pollinations, which
    generates a NEW image from the text prompt only (it does not edit the
    uploaded photo directly, just interprets its description via the prompt)."""
    import urllib.parse, random
    d = request.get_json(force=True) or {}
    prompt   = (d.get("prompt") or "").strip()
    ref_b64  = d.get("image")
    ref_mime = d.get("mime", "image/jpeg")

    if not prompt or not ref_b64:
        return jsonify({"error": "prompt and image required"}), 400

    try:
        encoded = urllib.parse.quote(prompt)
        seed = random.randint(1, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true&enhance=true&seed={seed}&nofeed=true"
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and resp.headers.get("content-type","").startswith("image/"):
            return jsonify({"image": base64.b64encode(resp.content).decode(), "mime": "image/png"})
    except Exception as e:
        print(f"[Pollinations fallback] {e}")

    return jsonify({"error": "Image edit failed. Please try again."}), 502


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
    providers_str = " -> ".join(active) if active else "none configured!"
    print(f"Starting Mythic AI at http://localhost:5000")
    print(f"Providers (fallback order): {providers_str}")
    app.run(host="0.0.0.0", port=5000, debug=False)
