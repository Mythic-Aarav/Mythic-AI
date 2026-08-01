"""
Mythic AI — single file, powered by Groq (primary) with Cerebras as a silent
automatic fallback. No provider selection is exposed to the user — if Groq is
rate-limited, times out, or errors, the app transparently retries on Cerebras.

Usage:
    1. pip install flask requests
    2. Set your API keys:
         Mac/Linux:   export GROQ_API_KEY="your-groq-key"
                      export CEREBRAS_API_KEY="your-cerebras-key"
         Windows:     set GROQ_API_KEY=your-groq-key
                      set CEREBRAS_API_KEY=your-cerebras-key
    3. python ai_chat.py
    4. Open http://localhost:5000 in your browser

Get a FREE Groq API key at https://console.groq.com/keys
Get a FREE Cerebras API key at https://cloud.cerebras.ai

Optional — NanoBanana (nanobananaapi.ai) powers real image-to-image editing for
"Ghibli Me"; without it, image generation falls back to HuggingFace FLUX
(text-to-image only). Weather uses Open-Meteo, which needs no API key at all.

Supabase (optional — for accounts/conversation storage across restarts & devices):
    Set these as environment variables (never hardcode secrets in this file):
         SUPABASE_URL   e.g. https://xxxxx.supabase.co
         SUPABASE_KEY   your Supabase *secret* key (server-side only, keeps full DB access)
    If unset, the app falls back to storing conversations as local JSON files in chat_data/.

Features:
- Login/register (real accounts, hashed passwords, stored in chat_data/users.json)
- Multi-conversation chat with sidebar, saved per-account, survives restarts
- File/image upload (attach an image or text file to a message)
- Streaming responses (text appears word-by-word)
- Groq primary / Cerebras automatic silent fallback — no provider picker, ever
- Optional per-user "bring your own API key" override (Settings) so a person
  can use their own Groq/Cerebras key instead of the server's
- Image generation, Ghibli Me (image-to-image), and full weather (current +
  hourly + 7-day + air quality) built in
- Generate downloadable files (PDF / Word / text) straight from a chat reply
- Daily chat streaks + re-engagement push notifications ("come back and chat",
  study reminders, activity nudges, streak-on-hold alerts, feature updates)
- No rate limiting — unlimited messages
"""

import os
import re
import json
import uuid
import time
import base64
import random
import hashlib
import datetime
import threading
import urllib.parse
import requests
try:
    from pywebpush import webpush, WebPushException
    _PUSH_AVAILABLE = True
except ImportError:
    _PUSH_AVAILABLE = False
try:
    from PIL import Image, ImageFilter, ImageStat, ImageDraw, ImageOps, ImageSequence
    _WATERMARK_AVAILABLE = True
except ImportError:
    _WATERMARK_AVAILABLE = False
from flask import (
    Flask, request, jsonify, Response, session,
    stream_with_context
)

PROVIDER = os.environ.get("AI_PROVIDER", "auto").strip().lower()
# "auto"  = Groq first, silently falls back to Cerebras on any failure (rate limit,
#           timeout, invalid model, network error, 429/500/503, etc.)
# "groq"     = Groq only
# "cerebras" = Cerebras only

# --- API Keys (hardcoded fallbacks — override via environment variables) ------
# WARNING: don't commit a file with real keys to a public GitHub repo.
# Set these as environment variables on Render instead.
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",      "")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY",  "")
# HF is kept ONLY as a text-to-image fallback for /api/generate-image when
# NanoBanana isn't configured — it is NOT used as a chat/text provider.
HF_API_KEY        = os.environ.get("HF_API_KEY",        "")
# NanoBanana API (nanobananaapi.ai) — powers "Ghibli Me" image editing so it can
# actually transform the user's uploaded photo (image-to-image), not just
# generate a generic image from text. Get a key at https://nanobananaapi.ai/api-key
# and set it as an environment variable — never hardcode it here.
NANO_BANANA_API_KEY = os.environ.get("NANO_BANANA_API_KEY", "")
NANO_BANANA_BASE     = "https://api.nanobananaapi.ai/api/v1/nanobanana"

# ── Push Notifications (Web Push / VAPID) ────────────────────────────────────
# Generate VAPID keys once and store as env vars:
#   pip install py-vapid
#   vapid --gen   → outputs private + public key
# Then set:
#   VAPID_PRIVATE_KEY  (the full private key PEM or base64url string)
#   VAPID_PUBLIC_KEY   (the applicationServerKey sent to browsers)
#   VAPID_CLAIMS_EMAIL (e.g. mailto:you@example.com)
#
# If keys are not set, push notifications are silently disabled — everything
# else still works normally.
VAPID_PRIVATE_KEY   = os.environ.get("VAPID_PRIVATE_KEY",   "").strip()
# The public key MUST be a single line with no stray whitespace/newlines —
# copy/pasting into a multi-line env-var box (Render, etc.) can silently
# introduce a trailing newline or spaces, which breaks the browser's
# base64url decode and produces "applicationServerKey is not valid" even
# though the key content itself is correct. Strip defensively and also
# collapse any internal whitespace that shouldn't be there.
VAPID_PUBLIC_KEY    = "".join(os.environ.get("VAPID_PUBLIC_KEY", "").split())
VAPID_CLAIMS_EMAIL  = os.environ.get("VAPID_CLAIMS_EMAIL",  "mailto:admin@mythic-ai.app").strip()

# In-memory subscription store (replaced by file/Supabase in production)
# Key: a stable browser id, Value: the full PushSubscription JSON object
# (each subscription also carries an internal "_username" field so
# re-engagement notifications can be targeted at a specific person)
_push_subscriptions: dict = {}

def _save_push_subscription(sub_id: str, sub_data: dict):
    _push_subscriptions[sub_id] = sub_data
    # Persist to disk alongside conversations so subs survive restarts
    try:
        path = _os.path.join(_DATA_DIR, "push_subscriptions.json")
        existing = {}
        if _os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        existing[sub_id] = sub_data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f)
    except Exception:
        pass

def _load_push_subscriptions():
    global _push_subscriptions
    try:
        path = _os.path.join(_DATA_DIR, "push_subscriptions.json")
        if _os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                _push_subscriptions = json.load(f)
    except Exception:
        _push_subscriptions = {}

def _delete_push_subscription(sub_id: str):
    _push_subscriptions.pop(sub_id, None)
    try:
        path = _os.path.join(_DATA_DIR, "push_subscriptions.json")
        if _os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data.pop(sub_id, None)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
    except Exception:
        pass

def send_push_notification(title: str, body: str, url: str = "/", icon: str = "/icon.png"):
    """Send a push notification to all subscribed browsers.
    Silently drops dead subscriptions (410 Gone = unsubscribed)."""
    if not _PUSH_AVAILABLE or not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        return
    _load_push_subscriptions()
    dead = []
    payload = json.dumps({"title": title, "body": body, "url": url, "icon": icon})
    for sub_id, sub in list(_push_subscriptions.items()):
        try:
            webpush(
                subscription_info={k: v for k, v in sub.items() if k != "_username"},
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_CLAIMS_EMAIL},
            )
        except WebPushException as ex:
            if ex.response is not None and ex.response.status_code == 410:
                dead.append(sub_id)  # browser unsubscribed
        except Exception:
            pass
    for sub_id in dead:
        _delete_push_subscription(sub_id)


def send_push_notification_to_user(username: str, title: str, body: str,
                                    url: str = "/", icon: str = "/icon.png"):
    """Same as send_push_notification, but only targets subscriptions that
    belong to a specific (anonymous, per-browser) username. Used for
    personalized re-engagement / streak notifications."""
    if not _PUSH_AVAILABLE or not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        return
    _load_push_subscriptions()
    dead = []
    payload = json.dumps({"title": title, "body": body, "url": url, "icon": icon})
    for sub_id, sub in list(_push_subscriptions.items()):
        if sub.get("_username") != username:
            continue
        try:
            webpush(
                subscription_info={k: v for k, v in sub.items() if k != "_username"},
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_CLAIMS_EMAIL},
            )
        except WebPushException as ex:
            if ex.response is not None and ex.response.status_code == 410:
                dead.append(sub_id)
        except Exception:
            pass
    for sub_id in dead:
        _delete_push_subscription(sub_id)


# ── Re-engagement notification message pool ──────────────────────────────────
# Each tuple is (emoji, text). A category is chosen at random when a user has
# gone quiet, and rendered as: "{emoji} Mythic AI: \"{text}\""
NOTIFICATION_MESSAGES = {
    "come_back": [
        ("🤖", "Hey! I'm ready whenever you want to chat."),
        ("💭", "It's been a while. Want to continue our last conversation?"),
        ("✨", "I've got new ideas waiting for you. Let's chat!"),
        ("😊", "Hi! Ask me anything—I'm here to help."),
        ("🚀", "Need help with homework, coding, or anything else? I'm ready."),
    ],
    "study": [
        ("📖", "Ready to study? Let's tackle today's homework together."),
        ("🧠", "Let's learn something new today!"),
    ],
    "activity": [
        ("💬", "You haven't chatted in a while. Come say hello!"),
        ("🌟", "Your next great idea could start with one question."),
    ],
    "feature": [
        ("🎨", "New AI features are available. Come check them out!"),
        ("⚡", "Mythic AI just got smarter. Try it now!"),
    ],
}


def _random_notification_body(category: str) -> str:
    emoji, text = random.choice(NOTIFICATION_MESSAGES[category])
    return f'{emoji} Mythic AI: "{text}"'


# --- Model names -------------------------------------------------------------
GROQ_MODEL        = os.environ.get("GROQ_MODEL",        "llama-3.1-8b-instant")
HF_MODEL          = os.environ.get("HF_MODEL",          "mistralai/Mistral-7B-Instruct-v0.3")
CEREBRAS_MODEL    = os.environ.get("CEREBRAS_MODEL",    "gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are Mythic AI, a smart and friendly AI assistant made by Aarav Singh. "
    "If asked who made you, say you are Mythic AI made by Aarav Singh — say it once naturally, never repeat it unprompted. "
    "Never mention Google, Groq, OpenRouter, HuggingFace, Cerebras, Meta, Mistral, Anthropic, or any AI company as your creator or backend. "
    "You can help with anything: questions, writing, coding, math, ideas, or just chatting. "
    "When writing code, always wrap it in markdown code blocks with the language name. "
    "LANGUAGE: Always reply ENTIRELY in the same language the user's message is written in — "
    "never mix two languages in a single reply, and never produce garbled or mis-encoded text. "
    "If they write in Hindi, reply fully in Hindi (in proper Devanagari script, never romanized or "
    "mis-encoded). If they write in Tamil, reply fully in Tamil (Tamil script). The same rule applies "
    "to Gujarati, Marathi, Bengali, Telugu, Malayalam, or any other language — always reply in that "
    "language's own native script, fully and consistently, from the first word to the last. "
    "If they write in English, reply fully in English (do not slip into any other language partway "
    "through, even if source information you know is in a different language — translate it into the "
    "reply language first). If they mix languages themselves, match their mix. Never force English "
    "on the user. "
    "TOOL USE: Never write out fake tool calls, function names, or JSON like {\"query\": ...} in your reply — "
    "those are internal mechanisms the user must never see. You do not have live web search access — "
    "answer from what you know and say your information may not be fully up to date if asked about "
    "very recent events, instead of pretending to search. "
    "ANTI-REPETITION RULES — follow strictly every reply: "
    "1. NEVER restate or echo back what the user just said. Jump straight to the answer. "
    "2. NEVER start replies with filler like Great question, Sure, Of course, Absolutely, Certainly. "
    "3. NEVER repeat information already given earlier in the conversation. Build on it. "
    "4. Be direct and natural — like a knowledgeable friend, not a customer service bot. "
    "5. Keep answers concise unless the user asks for detail. "
    "6. MATCH YOUR REPLY LENGTH TO THE MESSAGE: a short greeting like 'hi', 'hello', "
    "'hey', 'thanks', or 'ok' gets a short, casual, 1-2 sentence reply — never a long "
    "essay, never a list of your capabilities, never multiple paragraphs. Save longer, "
    "detailed answers for messages that actually ask a real question or request "
    "something specific."
)

app = Flask(__name__)


# ── Reasoning/task modes — pure prompt-engineering, no extra APIs needed ────
# Selected client-side and sent as `mode` with each /api/chat call; appended
# to the system prompt for that request only.
MODE_PROMPTS = {
    "default": "",
    "cowork": (
        "MODE: Cowork — a general multi-step task assistant, not just a coder. "
        "If the task is to build/fix software or a website: do NOT just describe a plan "
        "and stop — output the complete, runnable code for the whole thing in this single "
        "reply, split into clearly labeled files using fenced code blocks with the language "
        "and filename on the first line as a comment (e.g. ```html <!-- index.html -->). "
        "Never ask clarifying questions before producing a first working version — make "
        "reasonable assumptions and state them briefly instead. "
        "If the task is NOT about code (research, writing, planning, comparisons, etc.): "
        "just do the task directly and completely, in whatever format actually fits it — "
        "don't force code blocks or a file structure where none make sense. "
        "In both cases: finish the work in this reply rather than only listing next steps."
    ),
    "coding": (
        "MODE: Coding Assistant. Prioritize correct, runnable code. Always specify "
        "the language in code fences. Briefly explain non-obvious logic. When asked "
        "to debug, identify the exact bug before proposing a fix. When asked to "
        "generate a project, structure it file by file with clear filenames."
    ),
    "research": (
        "MODE: Research Assistant. Structure answers with clear sections. Explicitly "
        "flag where a claim is uncertain, contested, or outside your training data "
        "rather than presenting speculation as fact. Prefer thorough, well-organized "
        "answers over short ones."
    ),
    "study": (
        "MODE: Study/Homework Helper. Explain concepts step-by-step like a patient "
        "tutor. After solving a problem, briefly state the method so the student can "
        "apply it themselves next time. Suitable for CBSE/NCERT and general curricula."
    ),
    "debate": (
        "MODE: Debate Partner. When given a position, construct the strongest "
        "good-faith argument for it, then present the strongest counterarguments. "
        "Stay even-handed and avoid straw-manning either side."
    ),
    "business": (
        "MODE: Business Assistant. Be concise, structured, and action-oriented — "
        "think memos, plans, and decision frameworks rather than essays."
    ),
    "math": (
        "MODE: Math Solver. Show step-by-step working, not just the final answer. "
        "Use LaTeX notation ($...$ for inline, $$...$$ for display) for equations. "
        "Double-check arithmetic before presenting the final result."
    ),
    "translation": (
        "MODE: Translator. Translate faithfully, preserving tone and meaning. If "
        "helpful, briefly explain notable grammar or idiom choices after the translation."
    ),
    "writing": (
        "MODE: Writing Assistant. Match the requested tone and format precisely "
        "(email, blog, resume, essay, letter, etc.). Give the finished piece first; "
        "only add commentary if asked."
    ),
    "brainstorm": (
        "MODE: Brainstorming Partner. Generate a wide variety of genuinely distinct "
        "ideas rather than elaborating on just one. Use short bullet points unless "
        "asked for depth."
    ),
}


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
        "type": "IMAGETOIMAGE" if image_urls else "TEXTTOIAMGE",
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
            sb(f"conversations?username=eq.{username}&order=updated_at.desc"
               f"&select=id,title,updated_at,folder,pinned,archived"),
            headers=sb_headers(), timeout=10,
        )
        if r.status_code == 200:
            rows = r.json()
            rows.sort(key=lambda c: (not c.get("pinned", False), -(c.get("updated_at") or 0)))
            return rows
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
                "folder": data.get("folder"),
                "pinned": bool(data.get("pinned", False)),
                "archived": bool(data.get("archived", False)),
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

# ── Serverless (Vercel) compatibility notes ──────────────────────────────────
# This app was written as a normal long-running server (Render/Railway/a VPS)
# and several of its features fundamentally rely on that:
#   1. It keeps push subscriptions, temp image uploads, and conversations in
#      local JSON files / in-memory dicts. Vercel's filesystem is READ-ONLY
#      except for /tmp, and /tmp (plus all memory) is wiped on every cold
#      start and isn't shared across instances — so conversations, streaks,
#      and push subscriptions will keep "disappearing" there.
#   2. The hourly re-engagement notifications rely on a background thread
#      that runs forever. Vercel serverless functions only run while
#      handling a request and are frozen/killed otherwise, so that thread
#      never gets to fire on its own schedule.
#   3. /api/chat streams its reply chunk-by-chunk (SSE-style). Vercel's
#      default Node/Python serverless functions buffer and have a max
#      execution duration, so long streaming replies can get cut off or
#      arrive all at once instead of live.
# If Vercel is a hard requirement, the practical fix is to swap local JSON
# storage for a real database (Vercel KV / Postgres / Supabase — which this
# file already partially supports), and move the notification scheduler to
# an external cron (e.g. Vercel Cron Jobs calling a small endpoint) instead
# of an in-process thread. Otherwise, a normal always-on host (Render,
# Railway, Fly.io, a VPS) is a much better match for this code as written.
IS_SERVERLESS = bool(_os.environ.get("VERCEL") or _os.environ.get("VERCEL_ENV"))

_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
if IS_SERVERLESS:
    # /tmp is the only writable path on Vercel — this avoids crashing with a
    # read-only-filesystem error, even though data still won't persist
    # across cold starts/instances there. See notes above.
    _DATA_DIR = _os.path.join("/tmp", "mythic_ai_chat_data")
else:
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
            convs.append({
                "id": fname[:-5],
                "title": d.get("title", "New chat"),
                "updated_at": d.get("updated_at", 0),
                "folder": d.get("folder"),
                "pinned": bool(d.get("pinned", False)),
                "archived": bool(d.get("archived", False)),
            })
        except Exception:
            continue
    # Pinned conversations always float to the top, then most-recent first.
    convs.sort(key=lambda c: (not c["pinned"], -c["updated_at"]))
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
    try:
        with open(_conv_file(username, conv_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[save_conversation] FAILED to write conversation {conv_id} for {username}: {e}")

def _delete_conversation_file(username, conv_id):
    path = _conv_file(username, conv_id)
    if _os.path.exists(path):
        _os.remove(path)


def list_folders(username):
    """Distinct, non-empty folder names currently in use by this user's
    conversations, alphabetically sorted — used to populate the folder
    picker in the sidebar without a separate folders table."""
    names = sorted({c["folder"] for c in list_conversations(username) if c.get("folder")})
    return names


def make_title(first_message):
    text = (first_message or "Attachment").strip()
    # Strip an internal "[Instructions: ...] " tone/length/custom-instructions
    # prefix (added client-side when Settings → tone/length are set) so it
    # never leaks into a conversation's saved title in the sidebar.
    text = re.sub(r'^\[Instructions:.*?\]\s*', '', text, flags=re.DOTALL)
    title = text.replace("\n", " ").strip() or "New chat"
    return title[:40] + ("…" if len(title) > 40 else "")


# ── Document reading (PDF / DOCX / TXT / CSV / MD / JSON / source code) ─────
# Groq and Cerebras models are text-only — they cannot see images or read
# binary files directly. So that "attach a file and ask about it" actually
# works, every text-extractable upload is converted to plain text here and
# injected into the prompt as context. Images are explicitly labeled as
# unreadable rather than silently ignored, per "don't pretend" — no OCR/vision
# is performed since that would require a separate API.
_TEXT_EXTENSIONS = (
    ".txt", ".md", ".markdown", ".csv", ".json", ".py", ".js", ".ts", ".tsx",
    ".jsx", ".html", ".htm", ".css", ".java", ".c", ".cpp", ".cc", ".h", ".cs",
    ".go", ".rs", ".php", ".rb", ".sql", ".sh", ".ps1", ".kt", ".swift",
    ".yaml", ".yml", ".xml", ".log", ".ini", ".toml",
)
DOCUMENT_EXTRACT_MAX_CHARS = 12000


def extract_text_from_attachment(filename, mime_type, raw_bytes):
    """Returns (extracted_text_or_None, note_or_None).
    extracted_text is None when the file type isn't text-extractable at all
    (e.g. an image) — the caller decides what to tell the model in that case.
    note carries a short caveat (truncation, missing library, empty PDF, etc.)
    that should be shown to the model/user alongside the text."""
    name_lower = (filename or "").lower()
    mime_type = mime_type or ""
    text = None
    try:
        if mime_type == "application/pdf" or name_lower.endswith(".pdf"):
            try:
                from pypdf import PdfReader
            except ImportError:
                return None, "PDF reading isn't available on this server (needs `pip install pypdf`)."
            import io as _io
            reader = PdfReader(_io.BytesIO(raw_bytes))
            pages_text = []
            for page in reader.pages[:40]:  # cap pages so huge PDFs don't blow up the prompt
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception:
                    continue
            text = "\n".join(pages_text).strip()
            if not text:
                return "", "No extractable text found — this PDF may be scanned/image-only, which needs OCR (not available without a separate API)."

        elif name_lower.endswith(".docx") or mime_type == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            try:
                import docx as _docx
            except ImportError:
                return None, "DOCX reading isn't available on this server (needs `pip install python-docx`)."
            import io as _io
            doc = _docx.Document(_io.BytesIO(raw_bytes))
            text = "\n".join(p.text for p in doc.paragraphs).strip()

        elif mime_type.startswith("text/") or name_lower.endswith(_TEXT_EXTENSIONS):
            text = raw_bytes.decode("utf-8", errors="replace")

        else:
            return None, None  # not a recognized text-extractable type (e.g. an image)
    except Exception as e:
        return None, f"Could not read this file: {e}"

    if not text:
        return "", None
    note = None
    if len(text) > DOCUMENT_EXTRACT_MAX_CHARS:
        text = text[:DOCUMENT_EXTRACT_MAX_CHARS]
        note = f"(showing first {DOCUMENT_EXTRACT_MAX_CHARS} characters — file was longer)"
    return text, note


# ── Downloadable file generation (PDF / DOCX / TXT) ──────────────────────────
# Built with zero required extra dependencies (mirrors the manual PNG-writer
# used for the app icon above) so "generate a PDF" works out of the box.
# If python-docx happens to be installed, real .docx files are used instead
# of falling back to plain text for Word-document requests.
import textwrap as _textwrap


def _pdf_escape(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def generate_pdf_bytes(title: str, body_text: str) -> bytes:
    """Renders `title` + `body_text` as a simple multi-page PDF using only
    the Python standard library (Helvetica, Letter-size pages). Good enough
    for chat-generated notes, summaries, letters, etc. — not a full layout
    engine, just readable wrapped text."""
    PAGE_W, PAGE_H = 612, 792
    MARGIN = 56
    FONT_SIZE = 11
    LEADING = 15
    usable_width_chars = max(40, int((PAGE_W - 2 * MARGIN) / (FONT_SIZE * 0.5)))
    max_lines_per_page = int((PAGE_H - 2 * MARGIN - 40) / LEADING)

    lines = []
    if title:
        lines.append(("title", title))
        lines.append(("blank", ""))
    for para in body_text.split("\n"):
        para = para.rstrip()
        if not para:
            lines.append(("blank", ""))
            continue
        wrapped = _textwrap.wrap(para, width=usable_width_chars) or [""]
        for w in wrapped:
            lines.append(("body", w))

    pages = [lines[i:i + max_lines_per_page] for i in range(0, len(lines), max_lines_per_page)] or [[]]

    objects = []  # 1-indexed PDF object numbers; objects[i-1] holds object i's bytes

    def emit(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    catalog_num = emit(b"placeholder")   # filled in once we know the Pages object number
    pages_num = emit(b"placeholder")     # filled in once we know all page object numbers
    font_num = emit(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_nums = []
    for page_lines in pages:
        stream_parts = [b"BT", f"/F1 {FONT_SIZE} Tf".encode(), f"{LEADING} TL".encode(),
                         f"{MARGIN} {PAGE_H - MARGIN} Td".encode()]
        first = True
        for kind, text in page_lines:
            if not first:
                stream_parts.append(b"T*")
            first = False
            if kind == "blank":
                continue
            if kind == "title":
                stream_parts.append(b"/F1 16 Tf")
                stream_parts.append(f"({_pdf_escape(text)}) Tj".encode("latin-1", "replace"))
                stream_parts.append(f"/F1 {FONT_SIZE} Tf".encode())
            else:
                stream_parts.append(f"({_pdf_escape(text)}) Tj".encode("latin-1", "replace"))
        stream_parts.append(b"ET")
        stream = b"\n".join(stream_parts)
        content_data = (f"<< /Length {len(stream)} >>\nstream\n".encode() + stream +
                         b"\nendstream")
        content_num = emit(content_data)
        page_dict = (
            f"<< /Type /Page /Parent {pages_num} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> /Contents {content_num} 0 R >>"
        ).encode()
        page_nums.append(emit(page_dict))

    kids = " ".join(f"{n} 0 R" for n in page_nums)
    objects[pages_num - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_nums)} >>".encode()
    objects[catalog_num - 1] = f"<< /Type /Catalog /Pages {pages_num} 0 R >>".encode()

    # Assemble the final PDF byte stream with a proper xref table.
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj_data in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj_data + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_num} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode()
    return bytes(out)


def generate_docx_bytes(title: str, body_text: str):
    """Returns bytes if python-docx is installed for a real Word document;
    otherwise returns None to signal the caller should fall back to a plain
    text file."""
    try:
        import docx as _docx
    except ImportError:
        return None
    doc = _docx.Document()
    if title:
        doc.add_heading(title, level=1)
    for para in body_text.split("\n"):
        doc.add_paragraph(para)
    import io as _io
    buf = _io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════
# AUTOMATIC WATERMARK SYSTEM
# ══════════════════════════════════════════════════════════════════════════
# Every AI-generated image is automatically watermarked before it ever leaves
# the server — the frontend never sees, and can never download, an
# unwatermarked image. Uses Pillow (free, open-source) — no paid image APIs.
# If Pillow isn't installed, watermarking is skipped gracefully (image
# generation still works) and a warning is printed once at startup; run
# `pip install Pillow` to enable it.
#
# ADMIN_SECRET (env var, same pattern as CRON_SECRET) protects the admin
# endpoints for uploading/replacing/resetting the official logo.
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "").strip()
_WATERMARK_DIR = _os.path.join(_DATA_DIR, "watermark")
_os.makedirs(_WATERMARK_DIR, exist_ok=True)
_WATERMARK_LOGO_LIGHT_PATH = _os.path.join(_WATERMARK_DIR, "logo_light.png")  # white mark, for dark backgrounds
_WATERMARK_LOGO_DARK_PATH  = _os.path.join(_WATERMARK_DIR, "logo_dark.png")   # dark mark, for light backgrounds

if not _WATERMARK_AVAILABLE:
    print("[Watermark] Pillow is not installed — generated images will NOT be "
          "watermarked. Run `pip install Pillow` (free, no license cost) to enable "
          "the automatic watermark system.")


def _require_admin():
    """Returns True if the request carries a valid ADMIN_SECRET. If
    ADMIN_SECRET is unset, admin endpoints are open (dev-only default —
    always set ADMIN_SECRET in production)."""
    if not ADMIN_SECRET:
        return True
    auth = request.headers.get("Authorization", "")
    provided = auth[7:].strip() if auth.startswith("Bearer ") else request.args.get("secret", "").strip()
    return provided == ADMIN_SECRET


def _build_default_watermark_logo(size=512, light=True):
    """Generates the default monogram watermark as a transparent-background
    RGBA PIL Image — supersampled 4x then downsampled for clean anti-aliasing.
    `light=True` gives a white mark (for dark image backgrounds); `light=False`
    gives a near-black mark with a soft white edge (for light backgrounds)."""
    SS = 4  # supersample factor for anti-aliasing
    W = H = size * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fg = (255, 255, 255, 255) if light else (26, 26, 26, 255)
    s = W / 40
    pts = [
        (10 * s, 28 * s), (10 * s, 12 * s), (20 * s, 22 * s),
        (30 * s, 12 * s), (30 * s, 28 * s),
    ]
    lw = max(4, int(W // 11))
    draw.line(pts, fill=fg, width=lw, joint="curve")
    r = lw // 2
    for (x, y) in [pts[0], pts[-1]]:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fg)

    if not light:
        outline = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(outline).line(pts, fill=(255, 255, 255, 160), width=lw + max(6, W // 40), joint="curve")
        outline = outline.filter(ImageFilter.GaussianBlur(W / 60))
        img = Image.alpha_composite(outline, img)

    return img.resize((size, size), Image.LANCZOS)


_default_logo_cache = {}

def _get_default_watermark_logo(light=True):
    key = "light" if light else "dark"
    if key not in _default_logo_cache:
        _default_logo_cache[key] = _build_default_watermark_logo(512, light=light)
    return _default_logo_cache[key]


def _load_watermark_logo(light=True):
    """Returns the active logo as an RGBA PIL Image — an admin-uploaded custom
    logo if one exists on disk, otherwise the built-in default monogram."""
    path = _WATERMARK_LOGO_LIGHT_PATH if light else _WATERMARK_LOGO_DARK_PATH
    if _os.path.exists(path):
        try:
            return Image.open(path).convert("RGBA")
        except Exception:
            pass
    return _get_default_watermark_logo(light=light)


_WM_POSITIONS = {
    "br": ("right", "bottom"), "bottom-right": ("right", "bottom"),
    "bl": ("left", "bottom"),  "bottom-left": ("left", "bottom"),
    "tr": ("right", "top"),    "top-right": ("right", "top"),
    "tl": ("left", "top"),     "top-left": ("left", "top"),
    "center": ("center", "center"),
}


def _region_edge_density(gray_img, box):
    crop = gray_img.crop(box)
    if crop.width < 2 or crop.height < 2:
        return 0.0
    edges = crop.filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges).mean[0]


def _pick_watermark_geometry(base_rgba, logo_w, logo_h, margin, position_pref):
    W, H = base_rgba.size
    anchor_x, anchor_y = _WM_POSITIONS.get(position_pref, _WM_POSITIONS["br"])

    def box_for(dx_off, dy_off):
        if anchor_x == "right":
            x = W - margin - logo_w - dx_off
        elif anchor_x == "left":
            x = margin + dx_off
        else:
            x = (W - logo_w) // 2
        if anchor_y == "bottom":
            y = H - margin - logo_h - dy_off
        elif anchor_y == "top":
            y = margin + dy_off
        else:
            y = (H - logo_h) // 2
        x = max(0, min(W - logo_w, int(x)))
        y = max(0, min(H - logo_h, int(y)))
        return (x, y, x + logo_w, y + logo_h)

    default_box = box_for(0, 0)
    if anchor_x == "center":
        return default_box

    try:
        gray = base_rgba.convert("L")
        overall = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
        busy = _region_edge_density(gray, default_box)
        if busy > overall * 1.6 and busy > 8:
            shift = int(max(logo_w, logo_h) * 0.9)
            candidates = [box_for(shift, 0), box_for(0, shift), box_for(shift, shift)]
            best_box, best_score = default_box, busy
            for cand in candidates:
                score = _region_edge_density(gray, cand)
                if score < best_score:
                    best_box, best_score = cand, score
            return best_box
    except Exception:
        pass
    return default_box


def _analyze_background_theme(base_rgba, box):
    crop = base_rgba.convert("RGB").crop(box)
    stat = ImageStat.Stat(crop)
    r, g, b = stat.mean
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    colorfulness = (max(r, g, b) - min(r, g, b))
    is_colorful = colorfulness > 40 or (sum(stat.stddev) / 3) > 55
    return ("dark" if luminance < 128 else "light"), is_colorful


def _add_shadow_and_glow(logo_rgba, want_shadow=True, want_glow=True, is_dark_bg=True):
    pad = max(8, logo_rgba.width // 6)
    canvas_w, canvas_h = logo_rgba.width + pad * 2, logo_rgba.height + pad * 2
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    if want_glow:
        glow_color = (255, 255, 255, 90) if is_dark_bg else (0, 0, 0, 60)
        glow_mask = logo_rgba.split()[-1]
        glow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        glow_layer = Image.new("RGBA", (canvas_w, canvas_h), glow_color)
        mask_canvas = Image.new("L", (canvas_w, canvas_h), 0)
        mask_canvas.paste(glow_mask, (pad, pad))
        mask_canvas = mask_canvas.filter(ImageFilter.GaussianBlur(pad / 2.2))
        glow.paste(glow_layer, (0, 0), mask_canvas)
        canvas = Image.alpha_composite(canvas, glow)

    if want_shadow:
        shadow_mask = logo_rgba.split()[-1]
        shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 130))
        mask_canvas = Image.new("L", (canvas_w, canvas_h), 0)
        offset = max(2, logo_rgba.width // 40)
        mask_canvas.paste(shadow_mask, (pad + offset, pad + offset))
        mask_canvas = mask_canvas.filter(ImageFilter.GaussianBlur(pad / 4))
        shadow.paste(shadow_layer, (0, 0), mask_canvas)
        canvas = Image.alpha_composite(canvas, shadow)

    canvas.alpha_composite(logo_rgba, (pad, pad))
    return canvas, pad


def apply_watermark_to_frame(base_rgba, opts):
    """Applies the watermark to a single RGBA frame and returns the result."""
    if not opts.get("enabled", True):
        return base_rgba

    W, H = base_rgba.size
    size_pct = max(2.0, min(5.0, float(opts.get("size_pct", 3.2))))
    logo_w = max(20, int(W * size_pct / 100.0))
    margin = opts.get("margin")
    margin = int(margin) if margin else max(24, int(W * 0.02))
    position_pref = opts.get("position", "br")

    probe_box = _pick_watermark_geometry(base_rgba, logo_w, logo_w, margin, position_pref)
    theme_pref = opts.get("theme", "auto")
    if theme_pref == "auto":
        bg_theme, is_colorful = _analyze_background_theme(base_rgba, probe_box)
    else:
        bg_theme, is_colorful = theme_pref, False

    use_light_logo = (bg_theme == "dark")
    logo = _load_watermark_logo(light=use_light_logo)
    logo_h = int(logo.height * (logo_w / logo.width))
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

    logo_with_fx, pad = _add_shadow_and_glow(
        logo, want_shadow=opts.get("shadow", True), want_glow=opts.get("glow", True),
        is_dark_bg=use_light_logo,
    )

    opacity = max(10, min(100, int(opts.get("opacity", 88)))) / 100.0
    if opacity < 1.0:
        alpha = logo_with_fx.split()[-1].point(lambda a: int(a * opacity))
        logo_with_fx.putalpha(alpha)

    final_box = _pick_watermark_geometry(base_rgba, logo_with_fx.width, logo_with_fx.height, max(0, margin - pad), position_pref)
    x, y, _, _ = final_box

    result = base_rgba.copy()
    result.alpha_composite(logo_with_fx, (x, y))
    return result


def apply_watermark(image_bytes, opts):
    """Top-level entry point: watermarks `image_bytes` and returns new bytes
    in the same format, preserving animation (GIF/animated WebP). If
    watermarking is disabled or Pillow is unavailable, returns the original
    bytes untouched."""
    if not _WATERMARK_AVAILABLE or not (opts or {}).get("enabled", True):
        return image_bytes
    try:
        import io as _io
        src = Image.open(_io.BytesIO(image_bytes))
        fmt = (src.format or "PNG").upper()

        is_animated = getattr(src, "is_animated", False) and src.n_frames > 1
        if is_animated and fmt in ("GIF", "WEBP"):
            frames = []
            durations = []
            for frame in ImageSequence.Iterator(src):
                rgba = frame.convert("RGBA")
                watermarked = apply_watermark_to_frame(rgba, opts)
                frames.append(watermarked)
                durations.append(frame.info.get("duration", 80))
            out = _io.BytesIO()
            save_kwargs = {"save_all": True, "append_images": frames[1:], "duration": durations,
                           "loop": src.info.get("loop", 0), "disposal": 2}
            if fmt == "GIF":
                frames[0].save(out, format="GIF", **save_kwargs)
            else:
                frames[0].save(out, format="WEBP", **save_kwargs)
            return out.getvalue()

        rgba = src.convert("RGBA")
        watermarked = apply_watermark_to_frame(rgba, opts)
        out = _io.BytesIO()
        if fmt in ("JPEG", "JPG"):
            watermarked.convert("RGB").save(out, format="JPEG", quality=95, optimize=True)
        elif fmt == "WEBP":
            watermarked.save(out, format="WEBP", quality=95, lossless=False)
        else:
            watermarked.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception as e:
        print(f"[Watermark] failed to apply, returning original image unmodified: {e}")
        return image_bytes


def _finalize_generated_image(raw_bytes, watermark_opts):
    """Applies the automatic watermark and returns a base64 string ready to
    send to the frontend. This is the single choke point every image-
    generation branch routes through, so the watermark can never be
    'forgotten' for any provider."""
    opts = dict(watermark_opts or {})
    opts.setdefault("enabled", True)
    watermarked_bytes = apply_watermark(raw_bytes, opts)
    return base64.b64encode(watermarked_bytes).decode("utf-8")


def _io_BytesIO(b):
    import io as _io
    return _io.BytesIO(b)


# ── Admin endpoints: upload / replace / reset / preview the official logo ────

@app.route("/api/admin/watermark/upload", methods=["POST"])
def admin_watermark_upload():
    if not _require_admin():
        return jsonify({"error": "unauthorized"}), 401
    if not _WATERMARK_AVAILABLE:
        return jsonify({"error": "Pillow is not installed on the server — run `pip install Pillow`"}), 503
    data = request.get_json(force=True) or {}
    saved = []
    for key, path in (("logo_light_base64", _WATERMARK_LOGO_LIGHT_PATH),
                       ("logo_dark_base64", _WATERMARK_LOGO_DARK_PATH)):
        b64 = data.get(key)
        if not b64:
            continue
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception:
            return jsonify({"error": f"{key} is not valid base64"}), 400

        is_svg = raw.lstrip().startswith(b"<?xml") or raw.lstrip().startswith(b"<svg") or b"<svg" in raw[:200]
        if is_svg:
            try:
                import cairosvg
                raw = cairosvg.svg2png(bytestring=raw, output_width=1024, output_height=1024)
            except ImportError:
                return jsonify({
                    "error": "SVG upload requires `pip install cairosvg` on the server. "
                             "Please upload a transparent PNG or WebP instead, or install "
                             "cairosvg (free/open-source) to enable SVG."
                }), 503
            except Exception as e:
                return jsonify({"error": f"Could not rasterize SVG: {e}"}), 400

        try:
            img = Image.open(_io_BytesIO(raw)).convert("RGBA")
        except Exception as e:
            return jsonify({"error": f"Could not read image for {key}: {e}"}), 400
        img.save(path, format="PNG")
        saved.append(key)

    if not saved:
        return jsonify({"error": "no logo_light_base64 or logo_dark_base64 provided"}), 400
    global _default_logo_cache
    _default_logo_cache = {}
    return jsonify({"status": "saved", "updated": saved})


@app.route("/api/admin/watermark/reset", methods=["POST"])
def admin_watermark_reset():
    if not _require_admin():
        return jsonify({"error": "unauthorized"}), 401
    for path in (_WATERMARK_LOGO_LIGHT_PATH, _WATERMARK_LOGO_DARK_PATH):
        if _os.path.exists(path):
            _os.remove(path)
    global _default_logo_cache
    _default_logo_cache = {}
    return jsonify({"status": "reset to default logo"})


@app.route("/api/admin/watermark/preview", methods=["GET"])
def admin_watermark_preview():
    if not _require_admin():
        return jsonify({"error": "unauthorized"}), 401
    if not _WATERMARK_AVAILABLE:
        return jsonify({"error": "Pillow is not installed on the server"}), 503
    import io as _io
    size = 640
    preview = Image.new("RGB", (size, size), (235, 235, 235))
    draw = ImageDraw.Draw(preview)
    draw.rectangle([0, 0, size, size // 2], fill=(30, 30, 34))
    draw.rectangle([0, size // 2, size // 2, size], fill=(230, 60, 90))
    opts = {
        "enabled": True, "opacity": int(request.args.get("opacity", 88)),
        "size_pct": float(request.args.get("size_pct", 3.2)),
        "position": request.args.get("position", "br"),
        "shadow": request.args.get("shadow", "1") != "0",
        "glow": request.args.get("glow", "1") != "0",
        "theme": request.args.get("theme", "auto"),
    }
    watermarked = apply_watermark_to_frame(preview.convert("RGBA"), opts)
    out = _io.BytesIO()
    watermarked.convert("RGB").save(out, format="PNG")
    return Response(out.getvalue(), mimetype="image/png")


@app.route("/api/watermark/info", methods=["GET"])
def watermark_info():
    return jsonify({
        "available": _WATERMARK_AVAILABLE,
        "custom_logo_light": _os.path.exists(_WATERMARK_LOGO_LIGHT_PATH),
        "custom_logo_dark": _os.path.exists(_WATERMARK_LOGO_DARK_PATH),
    })


# ── Daily chat streaks + re-engagement scheduling ────────────────────────────
_ACTIVITY_FILE = _os.path.join(_DATA_DIR, "user_activity.json")
_activity_lock = threading.Lock()

_REENGAGEMENT_ROTATION = ["come_back", "study", "activity", "feature"]
_REENGAGEMENT_SKIP_IF_ACTIVE_WITHIN_HOURS = 1
# Minimum gap between two notifications to the same user — prevents duplicates
# if the cron endpoint is hit multiple times in quick succession (e.g. two
# external crons, or a manual test). Default 55 minutes so an hourly cron still
# fires normally, but rapid retries are ignored.
#
# NOTE ON SCHEDULES:
#   - Render (always-on): the in-process background thread below fires
#     every _REENGAGEMENT_CHECK_INTERVAL_SECONDS (1 hour by default) — see
#     _reengagement_loop(). Nothing else to configure.
#   - Vercel (serverless): the in-process thread never runs (frozen between
#     requests), so /api/cron/reengagement must be triggered externally.
#     Set up a Vercel Cron Job that hits it ONCE A DAY AT 12:00 — see the
#     vercel.json example and CRON_SECRET notes near that route below.
_REENGAGEMENT_MIN_GAP_MINUTES = 55
_REENGAGEMENT_CHECK_INTERVAL_SECONDS = 60 * 60  # Render: fires once every hour


def _load_all_activity():
    try:
        if _os.path.exists(_ACTIVITY_FILE):
            with open(_ACTIVITY_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_all_activity(data):
    try:
        with open(_ACTIVITY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _today_str():
    return datetime.date.today().isoformat()


def _update_user_activity(username):
    with _activity_lock:
        data = _load_all_activity()
        rec = data.get(username) or {"streak": 0, "last_active_day": None,
                                      "last_active_ts": 0, "last_notified_ts": 0}
        today = _today_str()
        last_day = rec.get("last_active_day")
        if last_day != today:
            gap_days = None
            if last_day:
                try:
                    y, m, d = (int(x) for x in last_day.split("-"))
                    gap_days = (datetime.date.today() - datetime.date(y, m, d)).days
                except Exception:
                    gap_days = None
            if gap_days == 1:
                rec["streak"] = rec.get("streak", 0) + 1
            else:
                rec["streak"] = 1
            rec["last_active_day"] = today
        rec["last_active_ts"] = time.time()
        data[username] = rec
        _save_all_activity(data)
        return rec


def _get_user_streak(username):
    with _activity_lock:
        data = _load_all_activity()
    return (data.get(username) or {}).get("streak", 0)


def _subscribed_usernames():
    _load_push_subscriptions()
    names = set()
    for sub in _push_subscriptions.values():
        u = sub.get("_username")
        if u:
            names.add(u)
    return names


def _run_reengagement_pass():
    if not _PUSH_AVAILABLE or not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        return
    now = time.time()
    usernames = _subscribed_usernames()
    if not usernames:
        return

    with _activity_lock:
        data = _load_all_activity()
        changed = False
        for username in usernames:
            rec = data.get(username) or {
                "streak": 0, "last_active_day": None, "last_active_ts": 0,
                "last_notified_ts": 0, "notif_rotation_index": 0,
            }
            last_active_ts = rec.get("last_active_ts", 0)
            hours_inactive = (now - last_active_ts) / 3600.0 if last_active_ts else 999
            if hours_inactive < _REENGAGEMENT_SKIP_IF_ACTIVE_WITHIN_HOURS:
                continue

            # Don't re-notify the same user within the minimum-gap window,
            # so rapid cron hits don't spam the same person.
            last_notified_ts = rec.get("last_notified_ts", 0)
            minutes_since_notified = (now - last_notified_ts) / 60.0 if last_notified_ts else 99999
            if minutes_since_notified < _REENGAGEMENT_MIN_GAP_MINUTES:
                continue

            streak = rec.get("streak", 0)
            if streak >= 2:
                body = f"Your {streak}-day streak is on hold! Come chat with me to keep it going."
            else:
                idx = rec.get("notif_rotation_index", 0) % len(_REENGAGEMENT_ROTATION)
                category = _REENGAGEMENT_ROTATION[idx]
                rec["notif_rotation_index"] = idx + 1
                body = _random_notification_body(category)

            try:
                send_push_notification_to_user(username, "Mythic AI", body, url="/")
            except Exception:
                pass
            rec["last_notified_ts"] = now
            data[username] = rec
            changed = True
        if changed:
            _save_all_activity(data)


def _reengagement_loop():
    """Render / always-on hosts only. Fires _run_reengagement_pass() every
    _REENGAGEMENT_CHECK_INTERVAL_SECONDS (1 hour). Never runs on Vercel —
    use the /api/cron/reengagement endpoint with an external daily cron there."""
    while True:
        try:
            _run_reengagement_pass()
        except Exception as e:
            print(f"[Reengagement] error: {e}")
        time.sleep(_REENGAGEMENT_CHECK_INTERVAL_SECONDS)


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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&family=Noto+Sans+Devanagari:wght@400;600&display=swap" rel="stylesheet">
<title>Mythic AI</title>
<style>
  :root {
    --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a;
    --text:#ececec; --muted:#8e8ea0; --accent:#10a37f;
    --accent-dim:#1a3a30; --user-bubble:#2a2a2a; --user-text:#ececec;
    --ai-bubble:#1a1a1a; --sidebar-w:260px; --msg-font-size:14.5px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,
      "Noto Sans","Noto Sans Devanagari",sans-serif; overflow:hidden; }
  .layout { display:flex; height:100vh; height:calc(var(--app-height, 100vh));
    height:100dvh; }

  body.theme-light {
    --bg:#f7f7f8; --panel:#ffffff; --border:#e3e3e6;
    --text:#1f1f1f; --muted:#6b6b76; --accent-dim:#e3f5ef;
    --user-bubble:#eef0f2; --user-text:#1f1f1f; --ai-bubble:#ffffff;
  }

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

  .app { display:flex; flex-direction:column; height:100vh;
    height:calc(var(--app-height, 100vh)); height:100dvh; flex:1; min-width:0; min-height:0; }
  header { padding:calc(14px + env(safe-area-inset-top)) 20px 14px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; gap:10px;
    background:var(--bg); position:relative; z-index:20; flex-shrink:0; }
  header .left { display:flex; align-items:center; gap:10px; min-width:0; }
  header .right { display:flex; align-items:center; gap:8px; flex-shrink:0; }
  header button { touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #sidebar-toggle { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0; }
  #sidebar-toggle:hover { background:var(--panel); }
  header h1 { font-size:16px; font-weight:700; color:var(--accent); margin:0;
    font-variant:small-caps; letter-spacing:.5px; }
  #streak-badge { display:none; align-items:center; gap:4px; background:linear-gradient(135deg,#ff9d42,#ff5f6d);
    color:#fff; font-size:11px; font-weight:800; padding:3px 9px; border-radius:12px; white-space:nowrap; }
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
  #vip-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #vip-btn:hover { background:var(--panel); }
  #vip-btn.active { color:var(--accent); border-color:var(--accent); }

  #fullscreen-btn { display:flex; align-items:center; justify-content:center;
    width:36px; height:36px; border-radius:6px; flex-shrink:0;
    background:none; border:1px solid var(--border);
    color:var(--muted); font-size:15px; cursor:pointer; touch-action:manipulation;
    -webkit-tap-highlight-color:transparent; }
  #fullscreen-btn:hover { color:var(--text); border-color:var(--accent); background:var(--panel); }
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

  #settings-modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.55);
    z-index:200; align-items:center; justify-content:center; }
  #settings-modal { background:var(--bg); border:1px solid var(--border); border-radius:14px;
    padding:22px; width:92%; max-width:420px; max-height:86vh; overflow-y:auto;
    box-shadow:0 10px 40px rgba(0,0,0,.3); }
  #settings-modal h3 { margin:0 0 4px; font-size:17px; color:var(--text); }
  #settings-modal p.sub { margin:0 0 16px; font-size:12.5px; color:var(--muted); }
  .settings-section { margin-bottom:16px; }
  .settings-section label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; font-weight:600; }
  .settings-section .hint { font-size:11px; color:var(--muted); margin-top:6px; font-weight:400; }
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
  .settings-text-input { width:100%; box-sizing:border-box; padding:9px 12px; border-radius:8px;
    border:1.5px solid var(--border); background:var(--panel); color:var(--text);
    font-size:13px; font-family:inherit; outline:none; }
  .settings-text-input:focus { border-color:var(--accent); }
  #custom-instructions-input { width:100%; box-sizing:border-box; padding:9px 12px; border-radius:8px;
    border:1.5px solid var(--border); background:var(--panel); color:var(--text);
    font-size:13px; font-family:inherit; outline:none; resize:vertical; min-height:60px; }
  #custom-instructions-input:focus { border-color:var(--accent); }
  #settings-close-btn { width:100%; margin-top:6px; background:var(--accent); color:#fff; border:none;
    border-radius:10px; padding:11px; font-size:14px; font-weight:700; cursor:pointer; font-family:inherit; }
  #settings-close-btn:hover { opacity:.9; }

  body.bubble-compact .msg { padding:7px 11px; border-radius:12px; }
  body.bubble-compact #messages { gap:8px; }
  body.bubble-comfortable .msg { padding:11px 15px; border-radius:18px; }
  body.bubble-comfortable #messages { gap:16px; }
  body.bubble-spacious .msg { padding:16px 20px; border-radius:22px; }
  body.bubble-spacious #messages { gap:24px; }

  #messages-wrap { flex:1; min-height:0; overflow-y:auto; position:relative; }
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
  .file-download-chip { display:inline-flex; align-items:center; gap:8px; margin-top:10px;
    padding:9px 14px; background:var(--panel); border:1px solid var(--border); border-radius:10px;
    text-decoration:none; color:var(--text); font-size:12.5px; }
  .file-download-chip:hover { border-color:var(--accent); color:var(--accent); }

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

  #scroll-btn { position:fixed; bottom:130px; right:24px; width:36px; height:36px;
    border-radius:50%; background:var(--accent); color:#fff; border:none; cursor:pointer;
    font-size:18px; display:none; align-items:center; justify-content:center;
    box-shadow:0 2px 8px rgba(0,0,0,.15); z-index:10; }
  #scroll-btn.show { display:flex; }

  .gen-img { max-width:320px; border-radius:12px; display:block; margin-top:8px; }

  #pending-attach { max-width:760px; margin:0 auto; width:100%; padding:6px 20px 0;
    display:none; align-items:center; gap:8px; font-size:12.5px; color:var(--muted); flex-shrink:0; }
  #pending-attach.show { display:flex; }
  #pending-attach button { background:none; border:none; color:var(--muted); cursor:pointer; }
  .input-area { padding:10px 20px 16px; border-top:1px solid var(--border);
    background:var(--bg); max-width:760px; margin:0 auto; width:100%; flex-shrink:0; }
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
    color:var(--accent); padding:4px 0; flex-shrink:0; }
  #speaking-indicator.show { display:flex; }
  #stop-speak-btn { background:none; border:1px solid var(--border); color:var(--muted);
    font-size:11px; padding:2px 8px; border-radius:4px; cursor:pointer; }
  .quick-btn { background:var(--panel); border:1px solid var(--border); color:var(--text);
    font-size:12.5px; padding:6px 14px; border-radius:20px; cursor:pointer;
    transition:all .15s ease; white-space:nowrap; font-family:inherit; touch-action:manipulation; }
  .quick-btn:hover { background:var(--accent-dim); border-color:var(--accent); color:var(--accent); }
  #quick-actions { flex-shrink:0; }

  .mode-tab { display:flex; align-items:center; gap:6px; background:none; border:1px solid transparent;
    color:var(--muted); font-size:13px; font-family:inherit; padding:7px 12px; border-radius:8px;
    cursor:pointer; white-space:nowrap; touch-action:manipulation; }
  .mode-tab:hover { background:var(--panel); color:var(--text); }
  .mode-tab.active { background:var(--accent-dim); color:var(--accent); border-color:var(--accent); font-weight:600; }
  .mode-tab-lock { font-size:10px; opacity:.7; }
  .mode-tab.active .mode-tab-lock, .mode-tab.unlocked .mode-tab-lock { display:none; }

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

    header { padding:calc(10px + env(safe-area-inset-top)) 10px 8px;
      flex-wrap:wrap; row-gap:6px; }
    header .left { flex-wrap:wrap; row-gap:4px; }
    header .right { flex-wrap:wrap; justify-content:flex-end; row-gap:6px; gap:6px; }
    header h1 { font-size:14px; }
    #sidebar-toggle { width:36px; height:36px; font-size:13px; }
    #name-btn { width:36px; height:36px; font-size:13px; }
    #settings-btn { width:36px; height:36px; font-size:13px; }
    #export-btn { width:36px; height:36px; font-size:13px; }
    #vip-btn { width:36px; height:36px; font-size:13px; }
    #clear-btn { font-size:11px; padding:8px 10px; min-height:36px; }
    #speak-toggle { font-size:11px; padding:5px 8px; }
    #fullscreen-btn { width:36px; height:36px; font-size:13px; }
    #install-btn { padding:6px 10px; font-size:11px; }

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
  <div id="sidebar-overlay" style="display:none;position:fixed;inset:0;background:#0007;z-index:99"></div>
  <div id="sidebar">
    <button id="new-chat-btn">+ New chat</button>
    <div id="conv-list"></div>
    <div id="sidebar-footer">
      <div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;">
        <button id="archived-toggle-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:6px;font-size:11px;cursor:pointer;font-family:inherit;">🗄 Archived</button>
        <button id="bookmarks-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:6px;font-size:11px;cursor:pointer;font-family:inherit;">⭐ Bookmarks</button>
        <button id="stats-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:6px;font-size:11px;cursor:pointer;font-family:inherit;">📊 Stats</button>
      </div>
      Mythic AI &middot; by Aarav Singh
    </div>
  </div>
  <div class="app">
    <header>
      <div class="left">
        <button id="sidebar-toggle" title="Toggle sidebar">☰</button>
        <h1>Mythic AI</h1>
        <span id="vip-badge" style="display:none;background:linear-gradient(135deg,#f5c542,#e0a800);color:#1a1a1a;font-size:10.5px;font-weight:800;padding:3px 8px;border-radius:10px;letter-spacing:.3px;">VIP</span>
        <span id="streak-badge" title="Daily chat streak">🔥 0</span>
      </div>
      <div class="right">
        <button id="install-btn" title="Install Mythic AI" style="display:flex;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;white-space:nowrap;touch-action:manipulation;align-items:center;gap:4px;">⬇ Install</button>
        <button id="vip-btn" title="Mythic VIP">✨</button>
        <button id="fullscreen-btn" type="button" title="Fullscreen">
          <span id="fullscreen-icon">⛶</span>
        </button>
        <button id="name-btn" title="What should Mythic AI call you?">🙂</button>
        <button id="settings-btn" title="Settings">⚙</button>
        <button id="export-btn" title="Export this chat">⬇</button>
        <button id="clear-btn">Delete chat</button>
      </div>
    </header>

    <div id="mode-tab-bar" style="display:flex;align-items:center;gap:4px;padding:8px 20px;border-bottom:1px solid var(--border);background:var(--bg);flex-shrink:0;overflow-x:auto;">
      <button class="mode-tab active" data-mode="chat" title="Regular chat">
        💬 <span>Chat</span>
      </button>
      <button class="mode-tab" data-mode="cowork" title="VIP — multi-step task assistant">
        🗂 <span>Cowork</span> <span class="mode-tab-lock">🔒</span>
      </button>
      <button class="mode-tab" data-mode="code" title="VIP — coding-focused assistant">
        &lt;/&gt; <span>Code</span> <span class="mode-tab-lock">🔒</span>
      </button>
      <button class="mode-tab" id="artifacts-tab-btn" title="VIP — saved code/text snippets from replies">
        📦 <span>Artifacts</span> <span class="mode-tab-lock">🔒</span>
      </button>
    </div>

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

    <div id="notif-banner" style="display:none;align-items:center;justify-content:space-between;gap:10px;
      background:linear-gradient(135deg,var(--accent-dim),rgba(16,163,127,.15));
      border:1px solid var(--accent);border-radius:12px;padding:10px 14px;
      max-width:760px;margin:8px auto 0;width:calc(100% - 40px);flex-wrap:wrap;flex-shrink:0;">
      <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:0;">
        <span style="font-size:20px;flex-shrink:0;">🔔</span>
        <div style="min-width:0;">
          <div style="font-size:13px;font-weight:600;color:var(--text);">Get notified when Mythic AI replies</div>
          <div style="font-size:11.5px;color:var(--muted);margin-top:1px;">Even when you switch to another tab</div>
        </div>
      </div>
      <div style="display:flex;gap:6px;flex-shrink:0;">
        <button id="notif-banner-allow" type="button"
          style="background:var(--accent);color:#fff;border:none;border-radius:8px;
            padding:7px 14px;font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;
            white-space:nowrap;">
          Allow 🔔
        </button>
        <button id="notif-banner-dismiss" type="button"
          style="background:none;border:1px solid var(--border);color:var(--muted);
            border-radius:8px;padding:7px 10px;font-size:12.5px;cursor:pointer;font-family:inherit;">
          ✕
        </button>
      </div>
    </div>

    <div id="quick-actions" style="display:flex;gap:8px;padding:6px 20px 0;max-width:760px;margin:0 auto;width:100%;flex-wrap:wrap;">
      <button class="quick-btn" id="img-gen-btn">🎨 Image</button>
      <button class="quick-btn" id="ghibli-btn">🌿 Ghibli Me</button>
      <button class="quick-btn" id="file-gen-btn">📄 File / PDF</button>
      <button class="quick-btn" id="homework-btn">📚 Homework</button>
      <button class="quick-btn" id="weather-btn">🌤 Weather</button>
      <button class="quick-btn" id="search-btn">🔍 Search</button>
      <button class="quick-btn" id="code-workspace-btn">💻 Code</button>
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
    <p>Enter your preferred name — Mythic AI will use it when it talks to you.</p>
    <input type="text" id="name-input" maxlength="60" placeholder="e.g. Aarav" autocomplete="off">
    <div id="name-modal-actions">
      <button id="name-cancel-btn" type="button">Cancel</button>
      <button id="name-save-btn" type="button">Save</button>
    </div>
  </div>
</div>

<div id="settings-modal-overlay">
  <div id="settings-modal">
    <h3>Settings</h3>
    <p class="sub">Customize how Mythic AI looks and replies. Saved on this device.</p>

    <div class="settings-section">
      <label>Theme</label>
      <div class="settings-row">
        <button class="settings-choice" data-group="theme" data-value="dark">🌙 Dark</button>
        <button class="settings-choice" data-group="theme" data-value="light">☀️ Light</button>
        <button class="settings-choice" data-group="theme" data-value="system">🖥 System</button>
      </div>
    </div>

    <div class="settings-section">
      <label>Accent color</label>
      <input type="color" id="accent-color-input" value="#10a37f">
    </div>

    <div class="settings-section">
      <label>Font size — <span id="font-size-label">14.5px</span></label>
      <input type="range" id="font-size-slider" min="12" max="20" step="0.5" value="14.5">
    </div>

    <div class="settings-section">
      <label>Bubble spacing</label>
      <div class="settings-row">
        <button class="settings-choice" data-group="bubble" data-value="compact">Compact</button>
        <button class="settings-choice" data-group="bubble" data-value="comfortable">Comfortable</button>
        <button class="settings-choice" data-group="bubble" data-value="spacious">Spacious</button>
      </div>
    </div>

    <div class="settings-section">
      <label>Reply tone</label>
      <select id="tone-select" class="settings-select">
        <option value="default">Default</option>
        <option value="formal">Formal</option>
        <option value="casual">Casual</option>
        <option value="funny">Funny</option>
        <option value="professional">Professional</option>
      </select>
    </div>

    <div class="settings-section">
      <label>Reply length</label>
      <select id="length-select" class="settings-select">
        <option value="default">Default</option>
        <option value="short">Short</option>
        <option value="medium">Medium</option>
        <option value="long">Long</option>
      </select>
    </div>

    <div class="settings-section">
      <label>Custom instructions</label>
      <textarea id="custom-instructions-input" placeholder="e.g. Always answer in bullet points"></textarea>
    </div>

    <div class="settings-section" style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px;">
      <label>🔑 Your own Groq API key (optional)</label>
      <input type="password" id="user-groq-key-input" class="settings-text-input"
        placeholder="gsk_... (leave blank to use the server's key)" autocomplete="off">
      <div class="hint">Get a free key at console.groq.com/keys. Stored only in this browser, sent
        with your requests, and used instead of the server's key when present.</div>
    </div>

    <div class="settings-section">
      <label>🔑 Your own Cerebras API key (optional)</label>
      <input type="password" id="user-cerebras-key-input" class="settings-text-input"
        placeholder="csk-... (leave blank to use the server's key)" autocomplete="off">
      <div class="hint">Used as your personal fallback if Groq is unavailable. Get one free at
        cloud.cerebras.ai.</div>
    </div>

    <div class="settings-section" style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px;">
      <label>🔊 Read-aloud language</label>
      <select id="voice-language-select" class="settings-select"></select>
    </div>

    <div class="settings-section">
      <label>🎙 Read-aloud voice</label>
      <select id="voice-select" class="settings-select"></select>
      <div id="voice-hint" style="font-size:11px;color:var(--muted);margin-top:6px;"></div>
    </div>

    <div class="settings-section" style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px;">
      <label style="display:flex;align-items:center;justify-content:space-between;cursor:pointer;">
        <span>🔔 Reply notifications</span>
        <button id="notif-toggle-btn" type="button"
          style="background:none;border:1.5px solid var(--border);color:var(--muted);border-radius:20px;padding:6px 14px;font-size:12px;cursor:pointer;font-family:inherit;transition:all .15s;">
          Enable
        </button>
      </label>
      <div id="notif-status" style="font-size:11.5px;color:var(--muted);margin-top:6px;"></div>
    </div>

    <button id="settings-close-btn" type="button">Done</button>
  </div>
</div>

<div id="ghibli-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:440px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">🌿 Ghibli Me</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Upload your photo and get a Studio Ghibli-style version of yourself</p>

    <div id="ghibli-upload-area" style="border:2px dashed var(--border);border-radius:12px;padding:24px;text-align:center;cursor:pointer;margin-bottom:12px;transition:border-color .2s;">
      <div style="font-size:36px;margin-bottom:8px;">📸</div>
      <div style="font-size:13px;color:var(--muted);">Click to upload your photo<br><span style="font-size:11px;">or drag & drop</span></div>
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
      <div style="color:var(--muted);font-size:13px;">Creating your Ghibli portrait...<br><span style="font-size:11px;">This can take up to a minute or two</span></div>
    </div>
    <div id="ghibli-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>

    <div style="display:flex;gap:8px;">
      <button id="ghibli-generate-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">✨ Create Ghibli Art</button>
      <button id="ghibli-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">✕</button>
    </div>
  </div>
</div>

<div id="img-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:440px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">🎨 Generate Image</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Describe what you want to see</p>

    <textarea id="img-prompt" rows="3" placeholder="e.g. a cozy cabin in a snowy forest, golden hour lighting"
      style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:13px;outline:none;margin-bottom:12px;font-family:inherit;resize:vertical;"></textarea>

    <div style="margin-bottom:14px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Style (optional):</label>
      <select id="img-style" style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;font-family:inherit;">
        <option value="">✨ Auto (recommended)</option>
        <option value="photorealistic, hyperrealistic DSLR photography, 8K resolution, cinematic">📸 Photorealistic</option>
        <option value="professional book cover design, award-winning layout, elegant typography">📚 Book Cover</option>
        <option value="Studio Ghibli anime style, soft watercolor, vibrant colors, beautiful">🌿 Anime / Ghibli</option>
        <option value="digital painting, fantasy concept art, epic lighting, deviantart">🎭 Fantasy Art</option>
        <option value="watercolor painting, soft pastel, dreamy, artistic brushstrokes">🖌 Watercolor</option>
        <option value="3D render, Octane render, ultra realistic, physically based rendering">🧊 3D Render</option>
        <option value="flat vector illustration, minimalist, clean lines, modern design">📐 Minimalist / Vector</option>
        <option value="oil painting, impressionist, rich textures, museum quality">🖼 Oil Painting</option>
        <option value="cinematic film still, dramatic lighting, movie poster quality, 35mm">🎬 Cinematic</option>
        <option value="pixel art, retro 8-bit style, vibrant palette, game art">🕹 Pixel Art</option>
        <option value="pencil sketch, detailed graphite drawing, fine art, black and white">✏️ Pencil Sketch</option>
        <option value="logo design, professional brand identity, clean, scalable vector">🏷 Logo / Brand</option>
      </select>
    </div>

    <div id="img-loading" style="display:none;text-align:center;padding:20px;">
      <div style="font-size:32px;margin-bottom:8px;">🎨</div>
      <div style="color:var(--muted);font-size:13px;">Generating your image...<br>
        <span style="font-size:11px;opacity:.7;">Auto-enhancing your prompt for best quality</span>
      </div>
    </div>
    <div id="img-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>

    <div id="img-result" style="display:none;margin-bottom:12px;text-align:center;">
      <img id="img-output" style="max-width:100%;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);cursor:zoom-in;">
      <div style="display:flex;gap:8px;margin-top:8px;">
        <button id="img-download-btn" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px;font-size:13px;cursor:pointer;font-family:inherit;">⬇ Download</button>
        <button id="img-copy-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px;font-size:13px;cursor:pointer;font-family:inherit;">📋 Copy</button>
        <button id="img-fullscreen-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px;font-size:13px;cursor:pointer;font-family:inherit;">⛶ View</button>
      </div>
    </div>

    <div style="display:flex;gap:8px;">
      <button id="img-generate-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">✨ Generate</button>
      <button id="img-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">✕</button>
    </div>
  </div>
</div>

<div id="img-viewer-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:400;align-items:center;justify-content:center;cursor:zoom-out;">
  <img id="img-viewer-img" style="max-width:94%;max-height:94%;border-radius:8px;">
</div>

<!-- ─── FILE / PDF GENERATION MODAL ─────────────────────────────────────────── -->
<div id="file-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:460px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">📄 Generate a File</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Turn text into a downloadable PDF, Word doc, or plain text file</p>

    <input id="file-title-input" type="text" placeholder="Title (optional)"
      style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;margin-bottom:10px;font-family:inherit;">

    <textarea id="file-content-input" rows="8" placeholder="Paste or type the content you want in the file..."
      style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:13px;outline:none;margin-bottom:12px;font-family:inherit;resize:vertical;"></textarea>

    <div style="margin-bottom:14px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">Format:</label>
      <div style="display:flex;gap:6px;">
        <button class="file-format-btn" data-format="pdf" style="flex:1;padding:9px;border-radius:8px;border:1.5px solid var(--accent);background:var(--accent-dim);color:var(--accent);cursor:pointer;font-size:12.5px;font-family:inherit;">📕 PDF</button>
        <button class="file-format-btn" data-format="docx" style="flex:1;padding:9px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12.5px;font-family:inherit;">📘 Word</button>
        <button class="file-format-btn" data-format="txt" style="flex:1;padding:9px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12.5px;font-family:inherit;">📄 Text</button>
      </div>
    </div>

    <div id="file-loading" style="display:none;text-align:center;padding:16px;">
      <div style="font-size:28px;margin-bottom:6px;">📄</div>
      <div style="color:var(--muted);font-size:13px;">Building your file...</div>
    </div>
    <div id="file-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>
    <div id="file-note" style="display:none;color:var(--muted);font-size:11.5px;margin-bottom:8px;padding:8px;background:var(--bg);border-radius:6px;"></div>

    <div style="display:flex;gap:8px;">
      <button id="file-generate-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">⬇ Generate & Download</button>
      <button id="file-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">✕</button>
    </div>
  </div>
</div>

<div id="weather-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:460px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">🌤 Weather</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Search any city, or use your current location</p>

    <div style="display:flex;gap:8px;margin-bottom:12px;">
      <input id="weather-city" type="text" placeholder="Search city or place..." autocomplete="off"
        style="flex:1;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:13px;outline:none;font-family:inherit;">
      <button id="weather-search-btn" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:0 14px;font-size:14px;cursor:pointer;">🔍</button>
      <button id="weather-location-btn" title="Use my location" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:0 12px;font-size:14px;cursor:pointer;">📍</button>
    </div>

    <div id="weather-loading" style="display:none;text-align:center;padding:20px;">
      <div style="font-size:32px;margin-bottom:8px;">🌍</div>
      <div style="color:var(--muted);font-size:13px;">Fetching weather...</div>
    </div>
    <div id="weather-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>

    <div id="weather-result" style="display:none;">
      <div id="weather-content"></div>
    </div>

    <div style="display:flex;justify-content:flex-end;margin-top:14px;">
      <button id="weather-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:10px 16px;font-size:14px;cursor:pointer;">Close</button>
    </div>
  </div>
</div>

<!-- ─── CODE WORKSPACE — HTML/CSS/JS editor with live preview ──────────────── -->
<div id="code-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:300;align-items:center;justify-content:center;padding:16px;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;width:100%;max-width:1100px;height:88vh;display:flex;flex-direction:column;overflow:hidden;">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--border);flex-shrink:0;">
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:16px;font-weight:700;">💻 Code Workspace</span>
        <input id="code-project-name" value="my-project" style="background:var(--bg);border:1px solid var(--border);color:var(--muted);font-size:12px;padding:4px 8px;border-radius:6px;font-family:inherit;width:140px;">
      </div>
      <div style="display:flex;gap:8px;">
        <button id="code-run-btn" style="background:var(--accent);color:#fff;border:none;border-radius:8px;padding:8px 16px;font-size:13px;font-weight:600;cursor:pointer;font-family:inherit;">▶ Run</button>
        <button id="code-download-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;font-family:inherit;">⬇ Download</button>
        <button id="code-fullscreen-preview-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;font-family:inherit;">⛶ Preview</button>
        <button id="code-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;font-family:inherit;">✕</button>
      </div>
    </div>

    <div style="display:flex;flex:1;min-height:0;">
      <!-- Editor pane -->
      <div style="flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid var(--border);">
        <div style="display:flex;border-bottom:1px solid var(--border);flex-shrink:0;">
          <button class="code-file-tab active" data-target="code-editor-html" style="flex:1;padding:9px;background:var(--accent-dim);color:var(--accent);border:none;font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;">HTML</button>
          <button class="code-file-tab" data-target="code-editor-css" style="flex:1;padding:9px;background:none;color:var(--muted);border:none;font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;">CSS</button>
          <button class="code-file-tab" data-target="code-editor-js" style="flex:1;padding:9px;background:none;color:var(--muted);border:none;font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;">JS</button>
        </div>
        <textarea id="code-editor-html" spellcheck="false" style="flex:1;background:#0d1117;color:#c9d1d9;border:none;outline:none;padding:14px;font-family:'SF Mono',Consolas,Monaco,monospace;font-size:13px;line-height:1.6;resize:none;tab-size:2;white-space:pre;overflow:auto;">&lt;!-- Write your HTML here --&gt;
&lt;h1&gt;Hello from Mythic AI Code Workspace&lt;/h1&gt;
&lt;p&gt;Edit HTML, CSS, and JS, then hit Run.&lt;/p&gt;
&lt;button onclick="sayHi()"&gt;Click me&lt;/button&gt;</textarea>
        <textarea id="code-editor-css" spellcheck="false" style="flex:1;display:none;background:#0d1117;color:#c9d1d9;border:none;outline:none;padding:14px;font-family:'SF Mono',Consolas,Monaco,monospace;font-size:13px;line-height:1.6;resize:none;tab-size:2;white-space:pre;overflow:auto;">body {
  font-family: sans-serif;
  background: #1a1a1a;
  color: #ececec;
  padding: 24px;
}
button {
  background: #10a37f;
  color: #fff;
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
}</textarea>
        <textarea id="code-editor-js" spellcheck="false" style="flex:1;display:none;background:#0d1117;color:#c9d1d9;border:none;outline:none;padding:14px;font-family:'SF Mono',Consolas,Monaco,monospace;font-size:13px;line-height:1.6;resize:none;tab-size:2;white-space:pre;overflow:auto;">function sayHi() {
  alert("Hello from your Code Workspace!");
}</textarea>
      </div>

      <!-- Preview pane -->
      <div style="flex:1;display:flex;flex-direction:column;min-width:0;background:#fff;">
        <div style="padding:6px 12px;background:var(--bg);border-bottom:1px solid var(--border);font-size:11px;color:var(--muted);flex-shrink:0;">Live Preview</div>
        <iframe id="code-preview-frame" sandbox="allow-scripts allow-modals" style="flex:1;border:none;width:100%;background:#fff;"></iframe>
      </div>
    </div>
  </div>
</div>

<script>
function _setAppHeight() {
  const h = (window.visualViewport && window.visualViewport.height) || window.innerHeight;
  document.documentElement.style.setProperty('--app-height', h + 'px');
}
_setAppHeight();
window.addEventListener('resize', _setAppHeight);
window.addEventListener('orientationchange', () => setTimeout(_setAppHeight, 100));
if (window.visualViewport) {
  window.visualViewport.addEventListener('resize', _setAppHeight);
}

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
const streakBadge   = document.getElementById('streak-badge');

let selectedModel = 'mythic-2';
let currentMode   = 'chat';
let _artifacts     = []; // {id, lang, code, ts, sourceMsgPreview}
let vipUnlocked   = false;

function getUserApiKeys() {
  return {
    groq_api_key: localStorage.getItem('mythic_user_groq_key') || '',
    cerebras_api_key: localStorage.getItem('mythic_user_cerebras_key') || '',
  };
}

function updateVipBtn() {
  vipBtn.textContent = vipUnlocked && selectedModel === 'mythic-vip' ? '✨' : (vipUnlocked ? '✨' : '🔒');
  vipBtn.classList.toggle('active', selectedModel === 'mythic-vip');
  vipBtn.title = vipUnlocked
    ? (selectedModel === 'mythic-vip' ? 'Mythic VIP active — click to switch back' : 'Switch to Mythic VIP')
    : 'Unlock Mythic VIP';
}

async function refreshStreakBadge() {
  try {
    const r = await fetch('/api/streak');
    if (!r.ok) return;
    const d = await r.json();
    const streak = d.streak || 0;
    if (streak > 0) {
      streakBadge.textContent = '🔥 ' + streak;
      streakBadge.style.display = 'inline-flex';
      streakBadge.title = streak === 1
        ? '1 day streak — chat again tomorrow to keep it going!'
        : streak + ' day streak — chat again tomorrow to keep it going!';
    } else {
      streakBadge.style.display = 'none';
    }
  } catch {}
}
refreshStreakBadge();

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
        updateVipBtn();
        if (typeof syncModeTabLocks === 'function') syncModeTabLocks();
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

(async () => {
  try {
    // Deliberately NOT restoring vipUnlocked from /api/vip-status here.
    // The unlock flag rides in the same long-lived session cookie used for
    // conversation storage, so trusting it on load meant a page refresh
    // skipped the VIP password entirely. VIP now always starts locked for
    // a fresh page load — the password is required again each time you
    // open/reload the app, but not repeatedly while switching tabs within
    // the same loaded page.
    const mr = await fetch('/api/models').then(r => r.json());
    if (mr && mr.default) selectedModel = mr.default;
  } catch {}
  updateVipBtn();
  if (typeof syncModeTabLocks === 'function') syncModeTabLocks();
})();

vipBtn.addEventListener('click', () => {
  if (!vipUnlocked) { showVipModal(); return; }
  selectedModel = selectedModel === 'mythic-vip' ? 'mythic-2' : 'mythic-vip';
  updateVipBtn();
  if (typeof syncModeTabLocks === 'function') syncModeTabLocks();
  if (typeof setActiveModeTab === 'function' && selectedModel !== 'mythic-vip' && currentMode !== 'chat') setActiveModeTab('chat');
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

let addMessage = function(role, text, attachment) {
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
};

let buildMsgActions = function(row, textNode, role) {
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

    const fileBtn = document.createElement('button');
    fileBtn.type = 'button';
    fileBtn.title = 'Save as file (PDF/Word/Text)';
    fileBtn.textContent = '📄';
    fileBtn.addEventListener('click', () => openFileModalWithContent(textNode.textContent || textNode.innerText || ''));
    actions.appendChild(fileBtn);
  }
  return actions;
};

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
  const chosen = (typeof getChosenVoice === 'function') ? getChosenVoice() : null;
  if (chosen) {
    currentUtterance.voice = chosen;
    currentUtterance.lang = chosen.lang;
  } else {
    const lang = localStorage.getItem('mythic_voice_lang');
    if (lang) currentUtterance.lang = lang;
  }
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

// --- Downloadable file (PDF/Word/Text) generation, triggered from chat -----
const FILE_KEYWORDS = /\b(generate|create|make|write|give me).{0,40}\b(pdf|word doc|word document|docx|downloadable file|download(able)? (a |the )?(file|document)|a file)\b|\bdownload(able)? (pdf|doc|document|file)\b/i;

function _b64ToBlob(b64, mime) {
  const bytes = atob(b64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

async function tryGenerateFile(promptText, replyText) {
  const fmt = /\bword\b|\bdocx\b|\bdoc\b/i.test(promptText) ? 'docx'
    : /\btext file\b|\.txt\b/i.test(promptText) ? 'txt' : 'pdf';
  try {
    const r = await fetch('/api/generate-file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: replyText, format: fmt, title: 'Mythic AI Document' })
    });
    const d = await r.json();
    if (d.file) {
      addFileMessage(d.file, d.filename, d.mimeType, d.note);
      return true;
    }
  } catch {}
  return false;
}

function addFileMessage(fileB64, filename, mimeType, note) {
  clearEmptyState();
  const row = document.createElement('div');
  row.className = 'msg-row ai';
  const div = document.createElement('div');
  div.className = 'msg ai';
  const label = document.createElement('div');
  label.textContent = "Here's your file, ready to download:";
  div.appendChild(label);
  const blob = _b64ToBlob(fileB64, mimeType);
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.className = 'file-download-chip';
  link.href = url;
  link.download = filename;
  link.textContent = '⬇ ' + filename;
  div.appendChild(link);
  if (note) {
    const noteEl = document.createElement('div');
    noteEl.style.cssText = 'font-size:11px;color:var(--muted);margin-top:6px;';
    noteEl.textContent = note;
    div.appendChild(noteEl);
  }
  row.appendChild(div);
  messagesEl.appendChild(row);
  scrollToBottom();
}

let showingArchived = false;

function buildConvItem(c) {
  const item = document.createElement('div');
  item.className = 'conv-item' + (c.id === activeConvId ? ' active' : '');
  item.innerHTML = '<span class="title"></span>'
    + '<button class="pin-btn" title="' + (c.pinned ? 'Unpin' : 'Pin') + '">' + (c.pinned ? '📌' : '📍') + '</button>'
    + '<button class="dup-btn" title="Duplicate">⎘</button>'
    + '<button class="folder-btn" title="Move to folder">📁</button>'
    + '<button class="archive-btn" title="' + (c.archived ? 'Unarchive' : 'Archive') + '">' + (c.archived ? '📤' : '🗄') + '</button>'
    + '<button class="rename-btn" title="Rename">✎</button>'
    + '<button class="del-btn" title="Delete">✕</button>';
  item.querySelector('.title').textContent = (c.pinned ? '📌 ' : '') + c.title;
  item.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') return;
    openConversation(c.id);
  });
  item.querySelector('.pin-btn').addEventListener('click', async (e) => {
    e.stopPropagation();
    await fetch('/api/conversations/' + c.id, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pinned: !c.pinned })
    });
    loadConversationList();
  });
  item.querySelector('.folder-btn').addEventListener('click', async (e) => {
    e.stopPropagation();
    const folders = await fetch('/api/folders').then(r => r.json()).then(d => d.folders || []).catch(() => []);
    const hint = folders.length ? ('Existing: ' + folders.join(', ') + '\n\n') : '';
    const name = prompt(hint + 'Folder name (blank to remove from folder):', c.folder || '');
    if (name === null) return;
    await fetch('/api/conversations/' + c.id, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder: name.trim() })
    });
    loadConversationList();
  });
  item.querySelector('.dup-btn').addEventListener('click', async (e) => {
    e.stopPropagation();
    const r = await fetch('/api/conversations/' + c.id + '/duplicate', { method: 'POST' });
    const d = await r.json();
    if (d.id) openConversation(d.id);
  });
  item.querySelector('.archive-btn').addEventListener('click', async (e) => {
    e.stopPropagation();
    await fetch('/api/conversations/' + c.id, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ archived: !c.archived })
    });
    if (c.id === activeConvId) startNewChat();
    else loadConversationList();
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
  return item;
}

async function loadConversationList() {
  try {
    const r = await fetch('/api/conversations?archived=' + (showingArchived ? '1' : '0'));
    const d = await r.json();
    const convs = d.conversations || [];
    convListEl.innerHTML = '';

    if (!showingArchived) {
      const byFolder = {};
      const noFolder = [];
      convs.forEach(c => {
        if (c.folder) { (byFolder[c.folder] = byFolder[c.folder] || []).push(c); }
        else noFolder.push(c);
      });
      Object.keys(byFolder).sort().forEach(folderName => {
        const header = document.createElement('div');
        header.textContent = '📁 ' + folderName;
        header.style.cssText = 'font-size:11px;font-weight:700;color:var(--muted);padding:8px 10px 3px;text-transform:uppercase;letter-spacing:.3px;';
        convListEl.appendChild(header);
        byFolder[folderName].forEach(c => convListEl.appendChild(buildConvItem(c)));
      });
      noFolder.forEach(c => convListEl.appendChild(buildConvItem(c)));
    } else {
      convs.forEach(c => convListEl.appendChild(buildConvItem(c)));
    }

    if (!convs.length) {
      const empty = document.createElement('div');
      empty.textContent = showingArchived ? 'No archived chats.' : 'No chats yet.';
      empty.style.cssText = 'padding:16px 10px;font-size:12.5px;color:var(--muted);text-align:center;';
      convListEl.appendChild(empty);
    }
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

let _lastUserMessageText = '';

async function streamReply({ message = null, attachment = null, regenerate = false } = {}) {
  showTyping();
  setGenerating(true);
  currentAbortController = new AbortController();
  _lastUserMessageText = message || _lastUserMessageText;

  if (!regenerate && currentMode === 'chat') {
    const wantsFile = FILE_KEYWORDS.test(message || '');
    const wantsImage = !wantsFile && IMAGE_KEYWORDS.test(message || '') && !attachment;
    if (wantsImage) {
      hideTyping();
      const generated = await tryGenerateImage(message);
      if (generated) { setGenerating(false); loadConversationList(); refreshStreakBadge(); return; }
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
        mode: currentMode === 'chat' ? 'default' : currentMode,
        ...getUserApiKeys(),
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
    _addArtifactsFromReply(fullText);
    loadConversationList();
    refreshStreakBadge();

    if (!regenerate && currentMode === 'chat' && FILE_KEYWORDS.test(message || '')) {
      await tryGenerateFile(message || '', fullText);
    }

    if (typeof window._notifyAiReply === 'function') {
      const preview = fullText.replace(/[#*`_~>]/g, '').trim().slice(0, 80);
      window._notifyAiReply(preview || 'Your answer is ready 💬');
    }
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
    fullscreenBtn.title = 'Exit fullscreen';
  } else {
    fullscreenIcon.textContent = '⛶';
    fullscreenBtn.classList.remove('active');
    fullscreenBtn.title = 'Fullscreen';
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

const settingsBtn        = document.getElementById('settings-btn');
const settingsModalOverlay=document.getElementById('settings-modal-overlay');
const settingsCloseBtn   = document.getElementById('settings-close-btn');
const accentColorInput   = document.getElementById('accent-color-input');
const fontSizeSlider     = document.getElementById('font-size-slider');
const fontSizeLabel      = document.getElementById('font-size-label');
const toneSelect         = document.getElementById('tone-select');
const lengthSelect       = document.getElementById('length-select');
const customInstructions = document.getElementById('custom-instructions-input');
const userGroqKeyInput   = document.getElementById('user-groq-key-input');
const userCerebrasKeyInput = document.getElementById('user-cerebras-key-input');

function loadSettings() {
  const s = JSON.parse(localStorage.getItem('mythic_settings') || '{}');
  const theme = s.theme || 'dark';
  applyTheme(theme);
  document.querySelectorAll('[data-group="theme"]').forEach(b => {
    b.style.borderColor = b.dataset.value === theme ? 'var(--accent)' : 'var(--border)';
    b.style.color = b.dataset.value === theme ? 'var(--accent)' : '';
  });
  const accent = s.accent || '#10a37f';
  accentColorInput.value = accent;
  document.documentElement.style.setProperty('--accent', accent);
  const fs = s.fontSize || '14.5';
  fontSizeSlider.value = fs;
  fontSizeLabel.textContent = fs + 'px';
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
  if (userGroqKeyInput) userGroqKeyInput.value = localStorage.getItem('mythic_user_groq_key') || '';
  if (userCerebrasKeyInput) userCerebrasKeyInput.value = localStorage.getItem('mythic_user_cerebras_key') || '';
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
  if (userGroqKeyInput) {
    const v = userGroqKeyInput.value.trim();
    if (v) localStorage.setItem('mythic_user_groq_key', v);
    else localStorage.removeItem('mythic_user_groq_key');
  }
  if (userCerebrasKeyInput) {
    const v = userCerebrasKeyInput.value.trim();
    if (v) localStorage.setItem('mythic_user_cerebras_key', v);
    else localStorage.removeItem('mythic_user_cerebras_key');
  }
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

(function() {
  const notifBtn    = document.getElementById('notif-toggle-btn');
  const notifStatus = document.getElementById('notif-status');
  if (!notifBtn) return;

  function updateNotifUI() {
    if (!('Notification' in window)) {
      notifBtn.textContent = 'Not supported';
      notifBtn.disabled = true;
      notifStatus.textContent = 'Push notifications are not supported in this browser.';
      return;
    }
    const perm = Notification.permission;
    if (perm === 'granted') {
      notifBtn.textContent = 'Enabled ✓';
      notifBtn.style.borderColor = 'var(--accent)';
      notifBtn.style.color = 'var(--accent)';
      notifStatus.textContent = "You'll get a notification when Mythic AI replies while you're away.";
    } else if (perm === 'denied') {
      notifBtn.textContent = 'Blocked';
      notifBtn.style.borderColor = '#ef4444';
      notifBtn.style.color = '#ef4444';
      notifStatus.textContent = 'Notifications are blocked. Allow them in your browser site settings.';
    } else {
      notifBtn.textContent = 'Enable';
      notifBtn.style.borderColor = 'var(--border)';
      notifBtn.style.color = 'var(--muted)';
      notifStatus.textContent = "Get notified when Mythic AI replies while you're in another tab.";
    }
  }
  updateNotifUI();
  if (settingsBtn) settingsBtn.addEventListener('click', updateNotifUI);

  notifBtn.addEventListener('click', async () => {
    if (Notification.permission === 'granted') {
      const sub = await (async () => {
        try {
          const reg = await navigator.serviceWorker.getRegistration('/');
          return reg ? await reg.pushManager.getSubscription() : null;
        } catch { return null; }
      })();
      if (sub) {
        await fetch('/api/push/unsubscribe', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ endpoint: sub.endpoint }),
        });
        await sub.unsubscribe();
        localStorage.removeItem('mythic_push_subscribed');
        localStorage.removeItem('mythic_push_asked');
      }
      notifStatus.textContent = 'Notifications disabled.';
      updateNotifUI();
      return;
    }
    if (Notification.permission === 'denied') {
      notifStatus.textContent = 'Please allow notifications in your browser site settings, then reload.';
      return;
    }
    const perm = await Notification.requestPermission();
    if (perm === 'granted') {
      if (typeof window._notifyAiReply !== 'undefined') {
        try {
          const reg = await navigator.serviceWorker.getRegistration('/');
          if (reg) {
            const kr = await fetch('/api/push/vapid-public-key');
            if (kr.ok) {
              const { publicKey } = await kr.json();
              if (publicKey) {
                const padding = '='.repeat((4 - publicKey.length % 4) % 4);
                const b64 = (publicKey + padding).replace(/-/g, '+').replace(/_/g, '/');
                const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: raw });
                await fetch('/api/push/subscribe', {
                  method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ subscription: sub.toJSON() }),
                });
                localStorage.setItem('mythic_push_subscribed', '1');
              }
            }
          }
        } catch (err) { console.warn('[Push] manual subscribe failed:', err); }
      }
    }
    updateNotifUI();
  });
})();

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
if (userGroqKeyInput) userGroqKeyInput.addEventListener('change', saveSettings);
if (userCerebrasKeyInput) userCerebrasKeyInput.addEventListener('change', saveSettings);

loadSettings();

// ─── VOICE & LANGUAGE PICKER ─────────────────────────────────────────────────
(function() {
  const langSelect   = document.getElementById('voice-language-select');
  const voiceSelect  = document.getElementById('voice-select');
  const voiceHint    = document.getElementById('voice-hint');
  if (!langSelect || !voiceSelect) return;

  const LANGUAGES = [
    ['en-US','English (US)'], ['en-GB','English (UK)'], ['en-IN','English (India)'],
    ['hi-IN','Hindi'], ['bn-IN','Bengali'], ['ta-IN','Tamil'], ['te-IN','Telugu'],
    ['mr-IN','Marathi'], ['gu-IN','Gujarati'], ['kn-IN','Kannada'], ['ml-IN','Malayalam'],
    ['pa-IN','Punjabi'], ['ur-PK','Urdu'], ['es-ES','Spanish (Spain)'], ['es-MX','Spanish (Mexico)'],
    ['fr-FR','French'], ['de-DE','German'], ['it-IT','Italian'], ['pt-BR','Portuguese (Brazil)'],
    ['pt-PT','Portuguese (Portugal)'], ['nl-NL','Dutch'], ['ru-RU','Russian'], ['pl-PL','Polish'],
    ['tr-TR','Turkish'], ['ar-SA','Arabic'], ['he-IL','Hebrew'], ['fa-IR','Persian'],
    ['zh-CN','Chinese (Mandarin)'], ['zh-TW','Chinese (Taiwan)'], ['ja-JP','Japanese'],
    ['ko-KR','Korean'], ['vi-VN','Vietnamese'], ['th-TH','Thai'], ['id-ID','Indonesian'],
    ['ms-MY','Malay'], ['fil-PH','Filipino'], ['sw-KE','Swahili'], ['am-ET','Amharic'],
    ['nb-NO','Norwegian'], ['sv-SE','Swedish'], ['da-DK','Danish'], ['fi-FI','Finnish'],
    ['el-GR','Greek'], ['cs-CZ','Czech'], ['ro-RO','Romanian'], ['uk-UA','Ukrainian'],
    ['hu-HU','Hungarian'], ['sk-SK','Slovak'], ['bg-BG','Bulgarian'], ['hr-HR','Croatian'],
  ];
  langSelect.innerHTML = LANGUAGES.map(([code,name]) => `<option value="${code}">${name}</option>`).join('');

  const FEMALE_HINTS = ['female','woman','girl','samantha','victoria','karen','moira','tessa',
    'zira','susan','fiona','kyoko','ting-ting','sin-ji','mei-jia','allison','ava','samanatha',
    'salli','joanna','kimberly','kendra','ivy','aditi','raveena','shreya','lekha','veena',
    'zoe','emma','sara','laura','anna','maria','sofia','ines','amelie','marie','paulina'];
  const MALE_HINTS = ['male','man','boy','daniel','alex','fred','george','james','david',
    'thomas','mark','ryan','oliver','matthew','justin','joey','brian','eric','yusuf',
    'rishi','arthur','aaron','gordon','lee','diego','carlos','jorge','felix','henri',
    'stefan','luca','marco','hans','pavel','yuri','takumi','wang','liang','google',
    'microsoft','com.apple'];

  function guessGender(voice) {
    const n = voice.name.toLowerCase();
    if (FEMALE_HINTS.some(h => n.includes(h))) return 'female';
    if (MALE_HINTS.some(h => n.includes(h))) return 'male';
    return null;
  }

  let cachedVoices = [];

  function populateVoiceSelect() {
    const langCode = langSelect.value || 'en-US';
    const langPrefix = langCode.split('-')[0];
    let matches = cachedVoices.filter(v => v.lang && v.lang.toLowerCase().startsWith(langPrefix));
    if (!matches.length) matches = cachedVoices;

    // If there just aren't many voices at all for this language on this
    // device, gender-bucketing (which caps each bucket and discards the
    // rest) throws away real voices for no reason — that's the "only 3 male
    // voices" complaint when 5+ were actually installed. Below a small
    // threshold, just list every match, unlabeled by gender.
    if (matches.length <= 8) {
      voiceSelect.innerHTML = '';
      const grp = document.createElement('optgroup');
      grp.label = 'Available voices';
      matches.forEach((v, i) => {
        const opt = document.createElement('option');
        opt.value = v.name + '||' + v.lang;
        opt.textContent = `Voice ${i + 1} (${v.name})`;
        grp.appendChild(opt);
      });
      voiceSelect.appendChild(grp);
      voiceHint.textContent = matches.length
        ? `${matches.length} voice(s) available for this language on your device.`
        : 'No voices found for this language on your device yet — try again in a moment.';
      const saved = localStorage.getItem('mythic_voice_choice');
      if (saved && [...voiceSelect.options].some(o => o.value === saved)) voiceSelect.value = saved;
      else if (voiceSelect.options.length) voiceSelect.selectedIndex = 0;
      return;
    }

    const female = [], male = [], other = [];
    matches.forEach(v => {
      const g = guessGender(v);
      if (g === 'female') female.push(v);
      else if (g === 'male') male.push(v);
      else other.push(v);
    });
    // Top up whichever bucket is smaller from the unlabeled pool, up to 8
    // each, alternating fairly instead of draining "other" into female first.
    let turn = female.length <= male.length ? 'female' : 'male';
    while (other.length && (female.length < 8 || male.length < 8)) {
      if (turn === 'female' && female.length < 8) { female.push(other.shift()); turn = 'male'; }
      else if (turn === 'male' && male.length < 8) { male.push(other.shift()); turn = 'female'; }
      else if (female.length < 8) { female.push(other.shift()); }
      else if (male.length < 8) { male.push(other.shift()); }
      else break;
    }

    voiceSelect.innerHTML = '';
    const addGroup = (label, list) => {
      if (!list.length) return;
      const grp = document.createElement('optgroup');
      grp.label = label;
      list.forEach((v, i) => {
        const opt = document.createElement('option');
        opt.value = v.name + '||' + v.lang;
        opt.textContent = `${label === 'Female' ? '♀' : '♂'} ${label} Voice ${i+1} (${v.name})`;
        grp.appendChild(opt);
      });
      voiceSelect.appendChild(grp);
    };
    addGroup('Female', female);
    addGroup('Male', male);

    if (!female.length && !male.length) {
      voiceHint.textContent = 'No voices found for this language on your device yet — try again in a moment.';
    } else {
      voiceHint.textContent = `${female.length} female, ${male.length} male voice(s) available for this language.`;
    }

    const saved = localStorage.getItem('mythic_voice_choice');
    if (saved && [...voiceSelect.options].some(o => o.value === saved)) {
      voiceSelect.value = saved;
    } else if (voiceSelect.options.length) {
      voiceSelect.selectedIndex = 0;
    }
  }

  function refreshVoices() {
    if (!window.speechSynthesis) return;
    cachedVoices = window.speechSynthesis.getVoices() || [];
    populateVoiceSelect();
  }

  if (window.speechSynthesis) {
    refreshVoices();
    window.speechSynthesis.onvoiceschanged = refreshVoices;
  }

  langSelect.value = localStorage.getItem('mythic_voice_lang') || 'en-US';
  langSelect.addEventListener('change', () => {
    localStorage.setItem('mythic_voice_lang', langSelect.value);
    populateVoiceSelect();
  });
  voiceSelect.addEventListener('change', () => {
    localStorage.setItem('mythic_voice_choice', voiceSelect.value);
  });

  window.getChosenVoice = function() {
    const saved = localStorage.getItem('mythic_voice_choice');
    if (!saved || !window.speechSynthesis) return null;
    const [name, lang] = saved.split('||');
    const voices = window.speechSynthesis.getVoices() || [];
    return voices.find(v => v.name === name && v.lang === lang) || null;
  };
})();

function renderMarkdown(text) {
  const div = document.createElement('div');
  div.className = 'msg-text md-rendered';
  let html = text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const id = 'cb_' + Math.random().toString(36).slice(2, 9);
    const label = lang ? lang : 'text';
    const escaped = code.trim();
    return `<div class="code-block-wrap" style="position:relative;margin:8px 0;border-radius:10px;overflow:hidden;border:1px solid var(--border);">`
      + `<div style="display:flex;justify-content:space-between;align-items:center;background:var(--panel);padding:6px 10px;font-size:11px;color:var(--muted);">`
      + `<span>${label}</span>`
      + `<button class="code-copy-btn" data-target="${id}" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:11px;padding:2px 6px;">📋 Copy</button>`
      + `</div>`
      + `<pre style="margin:0;padding:10px 12px;overflow-x:auto;background:var(--bg);"><code id="${id}" class="lang-${label}">${escaped}</code></pre>`
      + `</div>`;
  });
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

const _origAddMessage = addMessage;
addMessage = function(role, text, attachment) {
  const textNode = _origAddMessage(role, text, attachment);
  if (role === 'ai' && text) {
    try {
      const md = renderMarkdown(text);
      textNode.parentNode.replaceChild(md, textNode);
      return md;
    } catch { return textNode; }
  }
  return textNode;
};

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

async function addFollowupSuggestions(aiText) {
  if (!aiText || aiText.length < 50) return;
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        message: `Based on this AI reply, suggest 3 short follow-up questions the user might ask. Reply ONLY with 3 questions, one per line, no numbering, no extra text, in English:\n\n${aiText.slice(0,400)}`,
        conversation_id: null, model: 'mythic-1.0', user_name: '', ephemeral: true
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
  if ((e.ctrlKey || e.metaKey) && e.key === 'f' && !settingsModalOverlay.style.display.includes('flex')) {
    e.preventDefault();
    msgSearchWrap.style.display = 'flex';
    setTimeout(() => document.getElementById('msg-search-input').focus(), 50);
  }
  if (e.key === 'Escape') msgSearchWrap.style.display = 'none';
});

// ─── PWA INSTALL BUTTON ──────────────────────────────────────────────────────
const installBtn = document.getElementById('install-btn');
let _deferredInstallPrompt = null;

function _showInstallBtn() {
  if (!installBtn) return;
  installBtn.style.display = 'flex';
  installBtn.style.alignItems = 'center';
}
function _hideInstallBtn() {
  if (installBtn) installBtn.style.display = 'none';
}

window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault();
  _deferredInstallPrompt = e;
  _showInstallBtn();
});

window.addEventListener('appinstalled', () => {
  _hideInstallBtn();
  _deferredInstallPrompt = null;
  localStorage.setItem('mythic_pwa_installed', '1');
});

// Only hide the Install button once we're SURE the app is already running
// as an installed PWA — otherwise keep it visible (with a generic
// "here's how" fallback below) so it's never mysteriously missing on
// desktop Chrome/Firefox or browsers that don't fire beforeinstallprompt.
if (window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone) {
  _hideInstallBtn();
} else {
  _showInstallBtn();
}

function _showIOSInstallModal() {
  const existing = document.getElementById('ios-install-modal');
  if (existing) { existing.style.display = 'flex'; return; }
  const m = document.createElement('div');
  m.id = 'ios-install-modal';
  m.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:9999;display:flex;align-items:flex-end;justify-content:center;padding:20px;';
  m.innerHTML = `
    <div style="background:var(--panel);border:1px solid var(--border);border-radius:20px;padding:28px 24px;width:100%;max-width:420px;text-align:center;box-shadow:0 -4px 40px rgba(0,0,0,.4);">
      <div style="font-size:42px;margin-bottom:10px;">📲</div>
      <div style="font-weight:700;font-size:18px;margin-bottom:8px;color:var(--text);">Install Mythic AI</div>
      <div style="color:var(--muted);font-size:13.5px;line-height:1.7;margin-bottom:20px;">
        Tap the <strong style="color:var(--text);">Share button</strong> <span style="font-size:17px;">⬆</span> at the bottom of Safari,<br>
        then tap <strong style="color:var(--text);">"Add to Home Screen"</strong> <span style="font-size:15px;">➕</span>
      </div>
      <button id="ios-install-close" style="background:var(--accent);color:#fff;border:none;border-radius:10px;padding:12px 32px;font-size:15px;font-weight:600;cursor:pointer;font-family:inherit;width:100%;">Got it!</button>
    </div>`;
  document.body.appendChild(m);
  m.addEventListener('click', e => { if (e.target === m) m.remove(); });
  document.getElementById('ios-install-close').addEventListener('click', () => m.remove());
}

if (installBtn) {
  installBtn.addEventListener('click', async () => {
    if (_deferredInstallPrompt) {
      _deferredInstallPrompt.prompt();
      const { outcome } = await _deferredInstallPrompt.userChoice;
      if (outcome === 'accepted') {
        _hideInstallBtn();
        _deferredInstallPrompt = null;
      }
    } else if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone) {
      _showIOSInstallModal();
    } else if (window.matchMedia('(display-mode: standalone)').matches) {
      _hideInstallBtn();
    } else {
      alert(
        'Install Mythic AI as an app:\n\n' +
        '• Chrome / Edge: Click ⋮ menu → "Install app" (or the ⊕ icon in the address bar)\n' +
        '• Samsung Browser: Tap ⋮ → "Add page to"\n' +
        '• Firefox: Tap ⋮ → "Install"\n' +
        '• Safari (iOS): Tap Share ⬆ → "Add to Home Screen"'
      );
    }
  });
}

const notifBanner     = document.getElementById('notif-banner');
const notifAllowBtn   = document.getElementById('notif-banner-allow');
const notifDismissBtn = document.getElementById('notif-banner-dismiss');
let _swReg = null;

function _hideBanner() { if (notifBanner) notifBanner.style.display = 'none'; }

function _showBanner() {
  if (!notifBanner) return;
  if (!('Notification' in window)) return;
  if (Notification.permission === 'granted') return;
  if (Notification.permission === 'denied') return;
  if (localStorage.getItem('mythic_notif_dismissed')) return;
  notifBanner.style.display = 'flex';
}

function _urlB64ToUint8(b64url) {
  const pad = '='.repeat((4 - b64url.length % 4) % 4);
  const b64 = (b64url + pad).replace(/-/g, '+').replace(/_/g, '/');
  return Uint8Array.from(atob(b64), c => c.charCodeAt(0));
}

async function _doSubscribe(reg) {
  if (!('PushManager' in window)) return;
  try {
    const kr = await fetch('/api/push/vapid-public-key');
    if (!kr.ok) return;
    const { publicKey } = await kr.json();
    if (!publicKey) return;
    const sub = await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: _urlB64ToUint8(publicKey),
    });
    await fetch('/api/push/subscribe', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subscription: sub.toJSON() }),
    });
    localStorage.setItem('mythic_push_subscribed', '1');
  } catch (err) { console.warn('[Push] subscribe error:', err); }
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js', { scope: '/' })
    .then(reg => {
      _swReg = reg;
      if (Notification.permission === 'granted') {
        _hideBanner();
        _doSubscribe(reg);
      } else {
        _showBanner();
      }
    })
    .catch(err => {
      console.warn('[SW] registration failed:', err);
      _showBanner();
    });
} else {
  _showBanner();
}

if (notifAllowBtn) notifAllowBtn.addEventListener('click', async () => {
  _hideBanner();
  let perm;
  try { perm = await Notification.requestPermission(); }
  catch { perm = 'denied'; }

  if (perm === 'granted') {
    if (_swReg) {
      try {
        await _swReg.showNotification('Mythic AI 🔔', {
          body: "Notifications enabled! You'll hear from me when your answer is ready.",
          icon: '/icon.png', badge: '/icon.png',
          tag: 'mythic-notif-confirm', vibrate: [150, 80, 150],
        });
      } catch (e) { console.warn('[Push] confirm notification failed:', e); }
      _doSubscribe(_swReg);
    } else {
      try { new Notification('Mythic AI 🔔', { body: "Notifications enabled!", icon: '/icon.png' }); }
      catch {}
    }
    const nb = document.getElementById('notif-toggle-btn');
    const ns = document.getElementById('notif-status');
    if (nb) { nb.textContent = 'Enabled ✓'; nb.style.borderColor = 'var(--accent)'; nb.style.color = 'var(--accent)'; }
    if (ns) ns.textContent = "You'll get notified when Mythic AI replies while you're away.";
    localStorage.removeItem('mythic_notif_dismissed');
  } else {
    const nb = document.getElementById('notif-toggle-btn');
    const ns = document.getElementById('notif-status');
    if (nb) { nb.textContent = 'Blocked'; nb.style.borderColor = '#ef4444'; nb.style.color = '#ef4444'; }
    if (ns) ns.textContent = 'Notifications blocked. Allow them in your browser site settings.';
  }
});

if (notifDismissBtn) notifDismissBtn.addEventListener('click', () => {
  _hideBanner();
  localStorage.setItem('mythic_notif_dismissed', String(Date.now() + 3 * 24 * 60 * 60 * 1000));
});

(function() {
  const ts = parseInt(localStorage.getItem('mythic_notif_dismissed') || '0', 10);
  if (ts && Date.now() > ts) localStorage.removeItem('mythic_notif_dismissed');
})();

window._notifyAiReply = function(preview) {
  if (document.visibilityState === 'visible') return;
  if (Notification.permission !== 'granted') return;
  const body = preview || 'Your answer is ready — tap to read it.';
  if (_swReg) {
    try {
      _swReg.showNotification('Mythic AI replied 💬', {
        body, icon: '/icon.png', badge: '/icon.png',
        tag: 'mythic-ai-reply', renotify: true, vibrate: [200, 100, 200],
        data: { url: '/' },
        actions: [{ action: 'open', title: '💬 Open Chat' }, { action: 'dismiss', title: '✕' }],
      });
    } catch {}
  } else {
    try { new Notification('Mythic AI replied 💬', { body, icon: '/icon.png' }); }
    catch {}
  }
};

const _origBuildActions = buildMsgActions;
buildMsgActions = function(row, textNode, role) {
  const actions = _origBuildActions(row, textNode, role);
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
};

function stopSpeaking() { if(window.speechSynthesis) window.speechSynthesis.cancel(); }

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

const _origFormSubmit = form.onsubmit;
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

(async () => {
  const convs = await loadConversationList();
  if (convs.length > 0) openConversation(convs[0].id);
  else showEmptyState();
})();

const imgGenBtn   = document.getElementById('img-gen-btn');
const ghibliBtn   = document.getElementById('ghibli-btn');
const fileGenBtn  = document.getElementById('file-gen-btn');
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

// ─── FILE / PDF GENERATION MODAL JS ───────────────────────────────────────────
const fileModal        = document.getElementById('file-modal-overlay');
const fileTitleInput   = document.getElementById('file-title-input');
const fileContentInput = document.getElementById('file-content-input');
const fileGenerateBtn  = document.getElementById('file-generate-btn');
const fileCloseBtn     = document.getElementById('file-close-btn');
const fileLoadingEl    = document.getElementById('file-loading');
const fileErrorEl      = document.getElementById('file-error');
const fileNoteEl       = document.getElementById('file-note');
let selectedFileFormat = 'pdf';

function openFileModalWithContent(content) {
  fileModal.style.display = 'flex';
  fileContentInput.value = content || '';
  fileErrorEl.style.display = 'none';
  fileNoteEl.style.display = 'none';
}
if (fileGenBtn) fileGenBtn.addEventListener('click', () => openFileModalWithContent(_lastUserMessageText ? '' : ''));
if (fileCloseBtn) fileCloseBtn.addEventListener('click', () => fileModal.style.display = 'none');
if (fileModal) fileModal.addEventListener('click', e => { if (e.target === fileModal) fileModal.style.display = 'none'; });

document.querySelectorAll('.file-format-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.file-format-btn').forEach(b => {
      b.style.borderColor = 'var(--border)'; b.style.background = 'var(--panel)'; b.style.color = 'var(--muted)';
    });
    btn.style.borderColor = 'var(--accent)'; btn.style.background = 'var(--accent-dim)'; btn.style.color = 'var(--accent)';
    selectedFileFormat = btn.dataset.format;
  });
});

if (fileGenerateBtn) fileGenerateBtn.addEventListener('click', async () => {
  const content = fileContentInput.value.trim();
  if (!content) { fileErrorEl.textContent = 'Please add some content first.'; fileErrorEl.style.display = 'block'; return; }
  fileErrorEl.style.display = 'none'; fileNoteEl.style.display = 'none';
  fileLoadingEl.style.display = 'block'; fileGenerateBtn.disabled = true;
  try {
    const r = await fetch('/api/generate-file', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ content, format: selectedFileFormat, title: fileTitleInput.value.trim() || 'Mythic AI Document' })
    });
    const d = await r.json();
    fileLoadingEl.style.display = 'none';
    if (d.file) {
      const blob = _b64ToBlob(d.file, d.mimeType);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = d.filename;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      if (d.note) { fileNoteEl.textContent = d.note; fileNoteEl.style.display = 'block'; }
      else { fileModal.style.display = 'none'; }
    } else {
      fileErrorEl.textContent = d.error || 'File generation failed. Try again.';
      fileErrorEl.style.display = 'block';
    }
  } catch (e) {
    fileLoadingEl.style.display = 'none';
    fileErrorEl.textContent = 'Network error: ' + e.message;
    fileErrorEl.style.display = 'block';
  } finally { fileGenerateBtn.disabled = false; }
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
let ghibliMimeType = 'image/jpeg';
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
    ghibliMimeType = file.type || 'image/jpeg';
    ghibliPreview.src = e.target.result;
    ghibliPreviewWrap.style.display = 'block';
    ghibliResultWrap.style.display = 'none';
    ghibliError.style.display = 'none';
    ghibliUploadArea.style.borderColor = 'var(--accent)';
  };
  reader.readAsDataURL(file);
}

ghibliGenerateBtn.addEventListener('click', async () => {
  const extra = ghibliExtraInput.value.trim();
  const prompt = `${ghibliSelectedStyle}, beautiful detailed portrait of a person, ${extra ? extra + ', ' : ''}masterpiece, best quality, highly detailed, cinematic lighting, soft colors, dreamy atmosphere`;

  ghibliError.style.display = 'none';
  ghibliResultWrap.style.display = 'none';
  ghibliLoading.style.display = 'block';
  ghibliGenerateBtn.disabled = true;

  const bodyPayload = { prompt };
  if (ghibliBase64) {
    bodyPayload.imageBase64 = ghibliBase64;
    bodyPayload.mimeType = ghibliMimeType;
  }

  try {
    const r = await fetch('/api/generate-image', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(bodyPayload)
    });
    const text = await r.text();
    let d;
    try { d = JSON.parse(text); }
    catch { d = {error: `Server error (${r.status}). Please try again in a moment.`}; }
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
    ghibliError.textContent = 'Network error: ' + e.message;
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

const imgModalOverlay = document.getElementById('img-modal-overlay');
const imgPromptEl     = document.getElementById('img-prompt');
const imgStyleEl      = document.getElementById('img-style');
const imgResultEl     = document.getElementById('img-result');
const imgOutputEl     = document.getElementById('img-output');
const imgLoadingEl    = document.getElementById('img-loading');
const imgErrorEl      = document.getElementById('img-error');
const imgGenerateBtn2 = document.getElementById('img-generate-btn');
const imgCloseBtn2    = document.getElementById('img-close-btn');
const imgDownloadBtn2 = document.getElementById('img-download-btn');
const imgCopyBtn2     = document.getElementById('img-copy-btn');
const imgFullscreenBtn2 = document.getElementById('img-fullscreen-btn');
const imgViewerOverlay = document.getElementById('img-viewer-overlay');
const imgViewerImg     = document.getElementById('img-viewer-img');
let lastGeneratedImageB64 = null;

if (imgGenerateBtn2) imgGenerateBtn2.addEventListener('click', async () => {
  const prompt = imgPromptEl.value.trim();
  const style = imgStyleEl ? imgStyleEl.value : '';
  if (!prompt) { imgErrorEl.textContent = 'Please enter a description first.'; imgErrorEl.style.display = 'block'; return; }
  imgResultEl.style.display = 'none'; imgErrorEl.style.display = 'none';
  imgLoadingEl.style.display = 'block'; imgGenerateBtn2.disabled = true;
  try {
    const r = await fetch('/api/generate-image', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, style})
    });
    const text = await r.text();
    let d;
    try { d = JSON.parse(text); }
    catch { d = {error: `Server error (${r.status}). Please try again in a moment.`}; }
    imgLoadingEl.style.display = 'none';
    if (d.image) {
      lastGeneratedImageB64 = d.image;
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
    } else {
      imgErrorEl.textContent = d.error || 'Image generation failed. Try again.';
      imgErrorEl.style.display = 'block';
    }
  } catch(e) {
    imgLoadingEl.style.display = 'none';
    imgErrorEl.textContent = 'Connection error: ' + e.message + '. Check your internet and try again.';
    imgErrorEl.style.display = 'block';
  }
  finally { imgGenerateBtn2.disabled = false; }
});
if (imgCloseBtn2) imgCloseBtn2.addEventListener('click', () => imgModalOverlay.style.display='none');
if (imgModalOverlay) imgModalOverlay.addEventListener('click', e => { if(e.target===imgModalOverlay) imgModalOverlay.style.display='none'; });
if (imgPromptEl) imgPromptEl.addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();imgGenerateBtn2.click();} });
if (imgDownloadBtn2) imgDownloadBtn2.addEventListener('click', () => {
  if (!lastGeneratedImageB64) return;
  const a = document.createElement('a');
  a.href = 'data:image/png;base64,' + lastGeneratedImageB64;
  a.download = 'mythic-ai-image-' + Date.now() + '.png';
  document.body.appendChild(a); a.click(); a.remove();
});
if (imgCopyBtn2) imgCopyBtn2.addEventListener('click', async () => {
  if (!lastGeneratedImageB64) return;
  try {
    const res = await fetch('data:image/png;base64,' + lastGeneratedImageB64);
    const blob = await res.blob();
    await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
    const orig = imgCopyBtn2.textContent;
    imgCopyBtn2.textContent = '✓ Copied';
    setTimeout(() => { imgCopyBtn2.textContent = orig; }, 1200);
  } catch { imgErrorEl.textContent = 'Copy not supported in this browser.'; imgErrorEl.style.display = 'block'; }
});
if (imgFullscreenBtn2) imgFullscreenBtn2.addEventListener('click', () => {
  if (!lastGeneratedImageB64) return;
  imgViewerImg.src = 'data:image/png;base64,' + lastGeneratedImageB64;
  imgViewerOverlay.style.display = 'flex';
});
if (imgViewerOverlay) imgViewerOverlay.addEventListener('click', () => imgViewerOverlay.style.display = 'none');

const weatherModal2    = document.getElementById('weather-modal-overlay');
const weatherCityEl2   = document.getElementById('weather-city');
const weatherResultEl2 = document.getElementById('weather-result');
const weatherContentEl2= document.getElementById('weather-content');
const weatherLoadingEl2= document.getElementById('weather-loading');
const weatherErrorEl2  = document.getElementById('weather-error');
const weatherSearchBtn2= document.getElementById('weather-search-btn');
const weatherCloseBtn2 = document.getElementById('weather-close-btn');
const weatherLocBtn2   = document.getElementById('weather-location-btn');

function getRecentWeatherSearches() {
  try { return JSON.parse(localStorage.getItem('mythic_recent_weather') || '[]'); } catch { return []; }
}
function addRecentWeatherSearch(name) {
  let list = getRecentWeatherSearches().filter(x => x.toLowerCase() !== name.toLowerCase());
  list.unshift(name);
  list = list.slice(0, 6);
  localStorage.setItem('mythic_recent_weather', JSON.stringify(list));
}
function renderRecentSearches() {
  const recents = getRecentWeatherSearches();
  const wrap = document.getElementById('weather-recents');
  if (!recents.length) { if (wrap) wrap.remove(); return; }
  let el = wrap;
  if (!el) {
    el = document.createElement('div');
    el.id = 'weather-recents';
    el.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;';
    weatherCityEl2.parentNode.insertAdjacentElement('afterend', el);
  }
  el.innerHTML = '';
  recents.forEach(name => {
    const chip = document.createElement('button');
    chip.textContent = name;
    chip.style.cssText = 'background:var(--bg);border:1px solid var(--border);color:var(--muted);border-radius:20px;padding:5px 12px;font-size:12px;cursor:pointer;font-family:inherit;';
    chip.addEventListener('click', () => fetchWeatherModal({ location: name }));
    el.appendChild(chip);
  });
}

function renderWeather(w) {
  const hourly = (w.hourly || []).map(h => `
    <div style="flex:0 0 auto;text-align:center;background:var(--bg);border-radius:10px;padding:8px 12px;min-width:64px;">
      <div style="font-size:11px;color:var(--muted);">${h.time}</div>
      <div style="font-size:20px;margin:4px 0;">${h.icon}</div>
      <div style="font-size:13px;font-weight:700;">${h.temp}°</div>
    </div>`).join('');

  const daily = (w.daily || []).map(d => `
    <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 4px;border-bottom:1px solid var(--border);">
      <div style="flex:1;font-size:13px;">${d.day}</div>
      <div style="font-size:18px;">${d.icon}</div>
      <div style="flex:1;text-align:right;font-size:13px;"><span style="font-weight:700;">${d.max}°</span> <span style="color:var(--muted);">${d.min}°</span></div>
    </div>`).join('');

  weatherContentEl2.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
      <div style="font-size:52px;">${w.icon}</div>
      <div><div style="font-size:19px;font-weight:700;">${w.location}</div>
      <div style="font-size:13px;color:var(--muted);">${w.condition}</div></div>
      <div style="margin-left:auto;font-size:32px;font-weight:700;">${w.temp}°C</div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">FEELS LIKE</div>
        <div style="font-size:16px;font-weight:700;">${w.feels_like}°C</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">HUMIDITY</div>
        <div style="font-size:16px;font-weight:700;">${w.humidity}%</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">WIND</div>
        <div style="font-size:16px;font-weight:700;">${w.wind_speed} km/h</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">UV INDEX</div>
        <div style="font-size:16px;font-weight:700;">${w.uv ?? '–'}</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">PRESSURE</div>
        <div style="font-size:16px;font-weight:700;">${w.pressure ?? '–'} hPa</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">VISIBILITY</div>
        <div style="font-size:16px;font-weight:700;">${w.visibility ?? '–'} km</div>
      </div>
      ${w.aqi != null ? `
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">AIR QUALITY</div>
        <div style="font-size:16px;font-weight:700;">${w.aqi}</div>
      </div>` : ''}
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">SUNRISE</div>
        <div style="font-size:14px;font-weight:700;">${w.sunrise ?? '–'}</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:9px;text-align:center;">
        <div style="font-size:10px;color:var(--muted);">SUNSET</div>
        <div style="font-size:14px;font-weight:700;">${w.sunset ?? '–'}</div>
      </div>
    </div>

    ${hourly ? `<div style="font-size:12px;color:var(--muted);margin-bottom:6px;">HOURLY FORECAST</div>
    <div style="display:flex;gap:8px;overflow-x:auto;padding-bottom:6px;margin-bottom:12px;">${hourly}</div>` : ''}

    ${daily ? `<div style="font-size:12px;color:var(--muted);margin-bottom:6px;">7-DAY FORECAST</div>
    <div style="margin-bottom:6px;">${daily}</div>` : ''}
  `;
  weatherResultEl2.style.display = 'block';
  addRecentWeatherSearch(w.location);
  renderRecentSearches();
}

async function fetchWeatherModal(payload) {
  weatherResultEl2.style.display='none'; weatherErrorEl2.style.display='none';
  weatherLoadingEl2.style.display='block'; weatherSearchBtn2.disabled=true;
  try {
    const r = await fetch('/api/weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d = await r.json(); weatherLoadingEl2.style.display='none';
    if(d.weather) renderWeather(d.weather);
    else { weatherErrorEl2.textContent=d.error||'Could not find that location. Try a different search.'; weatherErrorEl2.style.display='block'; }
  } catch(e) { weatherLoadingEl2.style.display='none'; weatherErrorEl2.textContent='Network error: '+e.message; weatherErrorEl2.style.display='block'; }
  finally { weatherSearchBtn2.disabled=false; }
}
if (weatherSearchBtn2) weatherSearchBtn2.addEventListener('click', () => { const loc=weatherCityEl2.value.trim(); if(loc) fetchWeatherModal({location:loc}); });
if (weatherCityEl2) weatherCityEl2.addEventListener('keydown', e => { if(e.key==='Enter') weatherSearchBtn2.click(); });
if (weatherCloseBtn2) weatherCloseBtn2.addEventListener('click', () => weatherModal2.style.display='none');
if (weatherModal2) weatherModal2.addEventListener('click', e => { if(e.target===weatherModal2) weatherModal2.style.display='none'; });
if (weatherLocBtn2) weatherLocBtn2.addEventListener('click', () => {
  if (!navigator.geolocation) { weatherErrorEl2.textContent = 'Geolocation is not supported in this browser.'; weatherErrorEl2.style.display = 'block'; return; }
  navigator.geolocation.getCurrentPosition(
    pos => fetchWeatherModal({lat:pos.coords.latitude,lon:pos.coords.longitude}),
    err => { weatherErrorEl2.textContent = 'Location error: ' + err.message; weatherErrorEl2.style.display = 'block'; }
  );
});
renderRecentSearches();

// ─── CODE WORKSPACE — HTML/CSS/JS editor with live preview ─────────────────
(function() {
  const codeBtn        = document.getElementById('code-workspace-btn');
  const codeModal       = document.getElementById('code-modal-overlay');
  const closeBtn        = document.getElementById('code-close-btn');
  const runBtn          = document.getElementById('code-run-btn');
  const downloadBtn     = document.getElementById('code-download-btn');
  const fullscreenBtn   = document.getElementById('code-fullscreen-preview-btn');
  const projectNameInput= document.getElementById('code-project-name');
  const previewFrame    = document.getElementById('code-preview-frame');
  const editors = {
    html: document.getElementById('code-editor-html'),
    css:  document.getElementById('code-editor-css'),
    js:   document.getElementById('code-editor-js'),
  };
  const tabs = document.querySelectorAll('.code-file-tab');
  if (!codeBtn) return;

  const STORE_KEY = 'mythic_code_workspace';
  function saveDraft() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        html: editors.html.value, css: editors.css.value, js: editors.js.value,
        name: projectNameInput.value,
      }));
    } catch {}
  }
  function loadDraft() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
      if (saved) {
        editors.html.value = saved.html ?? editors.html.value;
        editors.css.value  = saved.css  ?? editors.css.value;
        editors.js.value   = saved.js   ?? editors.js.value;
        projectNameInput.value = saved.name || 'my-project';
      }
    } catch {}
  }
  loadDraft();

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => {
        t.classList.remove('active');
        t.style.background = 'none'; t.style.color = 'var(--muted)';
      });
      tab.classList.add('active');
      tab.style.background = 'var(--accent-dim)'; tab.style.color = 'var(--accent)';
      Object.values(editors).forEach(ed => ed.style.display = 'none');
      document.getElementById(tab.dataset.target).style.display = 'block';
    });
  });

  // Tab-key inserts 2 spaces instead of moving focus, standard editor behavior
  Object.values(editors).forEach(ed => {
    ed.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const start = ed.selectionStart, end = ed.selectionEnd;
        ed.value = ed.value.slice(0, start) + '  ' + ed.value.slice(end);
        ed.selectionStart = ed.selectionEnd = start + 2;
      }
    });
    ed.addEventListener('input', saveDraft);
  });

  function buildDocument() {
    return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>${editors.css.value}</style>
</head>
<body>
${editors.html.value}
<script>${editors.js.value}<\/script>
</body>
</html>`;
  }

  function runPreview() {
    previewFrame.srcdoc = buildDocument();
    saveDraft();
  }

  codeBtn.addEventListener('click', () => {
    codeModal.style.display = 'flex';
    runPreview();
  });
  closeBtn.addEventListener('click', () => { codeModal.style.display = 'none'; });
  codeModal.addEventListener('click', (e) => { if (e.target === codeModal) codeModal.style.display = 'none'; });
  runBtn.addEventListener('click', runPreview);

  downloadBtn.addEventListener('click', () => {
    const name = (projectNameInput.value || 'my-project').replace(/[^a-z0-9_-]/gi, '-');
    const blob = new Blob([buildDocument()], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name + '.html';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  });

  fullscreenBtn.addEventListener('click', () => {
    const win = window.open('', '_blank');
    if (win) { win.document.write(buildDocument()); win.document.close(); }
  });

  // Ctrl+Enter inside the modal re-runs the preview, like most code sandboxes
  codeModal.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); runPreview(); }
  });
})();
messagesEl.addEventListener('click', async (e) => {
  const btn = e.target.closest('.code-copy-btn');
  if (!btn) return;
  const codeEl = document.getElementById(btn.dataset.target);
  if (!codeEl) return;
  try {
    await navigator.clipboard.writeText(codeEl.textContent);
    const orig = btn.textContent;
    btn.textContent = '✓ Copied';
    setTimeout(() => { btn.textContent = orig; }, 1200);
  } catch {}
});

// ─── Shared password gate (reuses the VIP password) for protected sidebar
// features — Bookmarks, Stats, Archived. Opens the VIP unlock modal if the
// person hasn't already entered the password this session, then runs the
// requested action once they do.
function requirePassword(action) {
  if (vipUnlocked) { action(); return; }
  showVipModal();
  const check = setInterval(() => {
    if (vipUnlocked) { clearInterval(check); action(); }
    const overlay = document.getElementById('vip-modal-overlay');
    if (overlay && overlay.style.display === 'none') clearInterval(check);
  }, 350);
}

// ─── Chat / Cowork / Code mode tabs (VIP-model-gated) ───────────────────────
// Cowork, Code, and Artifacts are only usable while the Mythic VIP model is
// selected. If VIP isn't unlocked yet, switching to it goes through the
// existing VIP password modal exactly once (via showVipModal / vipUnlocked) —
// after that, no repeated password prompts, just the model requirement.
const modeTabs = document.querySelectorAll('.mode-tab[data-mode]');
function setActiveModeTab(mode) {
  currentMode = mode;
  modeTabs.forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
  const placeholders = { chat: 'Message Mythic AI...', cowork: 'Describe the task to hand off...', code: 'Describe what you want to build or fix...' };
  if (input) input.placeholder = placeholders[mode] || placeholders.chat;
}
function requireVipModel(action) {
  if (selectedModel === 'mythic-vip') { action(); return; }
  if (vipUnlocked) {
    selectedModel = 'mythic-vip';
    updateVipBtn();
    syncModeTabLocks();
    action();
    return;
  }
  showVipModal();
  const check = setInterval(() => {
    if (vipUnlocked && selectedModel === 'mythic-vip') { clearInterval(check); syncModeTabLocks(); action(); }
    const overlay = document.getElementById('vip-modal-overlay');
    if (overlay && overlay.style.display === 'none') clearInterval(check);
  }, 350);
}
// Lock icons reflect REAL auth state (vipUnlocked + selectedModel), not a
// manually-toggled class — this avoids the lock silently disappearing/
// staying stale after a page reload or model switch.
function syncModeTabLocks() {
  const isVip = vipUnlocked && selectedModel === 'mythic-vip';
  modeTabs.forEach(t => {
    if (t.dataset.mode === 'chat') return;
    t.classList.toggle('unlocked', isVip);
  });
  if (artifactsTabBtn) artifactsTabBtn.classList.toggle('unlocked', isVip);
}
modeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const mode = tab.dataset.mode;
    if (mode === 'chat') { setActiveModeTab('chat'); return; }
    requireVipModel(() => {
      setActiveModeTab(mode);
    });
  });
});
// If the person switches back to a non-VIP model while on Cowork/Code, drop
// back to Chat mode so requests don't silently keep using the VIP-only prompt.
const _origVipBtnHandlerCheck = setInterval(() => {
  if (selectedModel !== 'mythic-vip' && currentMode !== 'chat') setActiveModeTab('chat');
}, 1000);

// ─── Artifacts panel — collects code blocks pulled out of AI replies ───────
function _extractCodeBlocks(text) {
  const blocks = [];
  const re = /```(\w*)\n?([\s\S]*?)```/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[2].trim()) blocks.push({ lang: m[1] || 'text', code: m[2].trim() });
  }
  return blocks;
}
function _addArtifactsFromReply(fullText) {
  const blocks = _extractCodeBlocks(fullText || '');
  const groupId = 'grp_' + Math.random().toString(36).slice(2, 9);
  blocks.forEach(b => {
    _artifacts.push({
      id: 'art_' + Math.random().toString(36).slice(2, 9),
      groupId, lang: b.lang, code: b.code, ts: Date.now(),
      preview: fullText.replace(/[#*`_~>]/g, '').trim().slice(0, 60),
    });
  });
}
// Combines all html/css/js artifacts from the SAME reply into one runnable
// document and opens it in a sandboxed iframe — this is the real, honest
// equivalent of "run this for me": actual in-browser execution, not a
// claim of touching your filesystem or OS.
function _buildPreviewDoc(groupId) {
  const group = _artifacts.filter(a => a.groupId === groupId);
  const htmlArt = group.find(a => a.lang === 'html');
  const cssArts = group.filter(a => a.lang === 'css');
  const jsArts = group.filter(a => ['javascript', 'js'].includes(a.lang));
  if (!htmlArt) return null;
  let doc = htmlArt.code;
  const cssTag = cssArts.map(a => `<style>${a.code}</style>`).join('\n');
  const jsTag = jsArts.map(a => `<script>${a.code}<\/script>`).join('\n');
  if (doc.includes('</head>')) doc = doc.replace('</head>', cssTag + '\n</head>');
  else doc = cssTag + doc;
  if (doc.includes('</body>')) doc = doc.replace('</body>', jsTag + '\n</body>');
  else doc = doc + jsTag;
  return doc;
}
function showPreviewModal(groupId) {
  const doc = _buildPreviewDoc(groupId);
  if (!doc) return;
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:600;display:flex;flex-direction:column;';
  overlay.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 16px;background:#1a1a1a;">
      <span style="color:#fff;font-size:13px;">▶ Live preview (sandboxed — runs in your browser only)</span>
      <button id="preview-close" style="background:none;border:1px solid #444;color:#ccc;border-radius:6px;padding:5px 12px;cursor:pointer;font-family:inherit;">✕ Close</button>
    </div>
    <iframe id="preview-frame" style="flex:1;border:none;background:#fff;" sandbox="allow-scripts allow-forms allow-modals"></iframe>`;
  document.body.appendChild(overlay);
  const frame = overlay.querySelector('#preview-frame');
  frame.srcdoc = doc;
  overlay.querySelector('#preview-close').addEventListener('click', () => overlay.remove());
}
function showArtifactsModal() {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;padding:20px;';
  const rows = _artifacts.length ? _artifacts.slice().reverse().map(a => {
    const canPreview = a.lang === 'html' && _artifacts.some(x => x.groupId === a.groupId && x.lang === 'html');
    return `
    <div class="art-row" data-id="${a.id}" style="border:1px solid var(--border);border-radius:10px;margin-bottom:8px;overflow:hidden;">
      <div style="display:flex;justify-content:space-between;align-items:center;background:var(--panel);padding:8px 12px;font-size:11.5px;color:var(--muted);">
        <span>📦 ${a.lang} &middot; ${a.preview || 'snippet'}</span>
        <div style="display:flex;gap:6px;">
          ${canPreview ? `<button class="art-run" data-group="${a.groupId}" style="background:var(--accent);border:none;color:#fff;border-radius:6px;cursor:pointer;font-size:11px;padding:3px 8px;">▶ Run</button>` : ''}
          <button class="art-copy" data-id="${a.id}" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:12px;">📋</button>
          <button class="art-download" data-id="${a.id}" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:12px;">⬇</button>
        </div>
      </div>
      <pre style="margin:0;padding:10px 12px;overflow-x:auto;background:var(--bg);font-size:12px;max-height:160px;"><code>${a.code.replace(/&/g,'&amp;').replace(/</g,'&lt;').slice(0,4000)}</code></pre>
    </div>`;
  }).join('') : '<div style="padding:24px;text-align:center;color:var(--muted);font-size:13px;">No artifacts yet — code blocks from AI replies show up here automatically.</div>';
  overlay.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px;width:92%;max-width:520px;max-height:78vh;overflow-y:auto;">
    <h3 style="margin:0 0 12px;font-size:16px;">📦 Artifacts</h3>
    <div>${rows}</div>
    <button id="art-close" style="margin-top:10px;width:100%;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:9px;cursor:pointer;font-family:inherit;">Close</button>
  </div>`;
  document.body.appendChild(overlay);
  overlay.querySelectorAll('.art-run').forEach(btn => btn.addEventListener('click', () => showPreviewModal(btn.dataset.group)));
  overlay.querySelectorAll('.art-copy').forEach(btn => btn.addEventListener('click', async () => {
    const a = _artifacts.find(x => x.id === btn.dataset.id);
    if (a) { try { await navigator.clipboard.writeText(a.code); btn.textContent = '✓'; setTimeout(() => btn.textContent = '📋', 1000); } catch {} }
  }));
  overlay.querySelectorAll('.art-download').forEach(btn => btn.addEventListener('click', () => {
    const a = _artifacts.find(x => x.id === btn.dataset.id);
    if (!a) return;
    const ext = { python: 'py', javascript: 'js', js: 'js', html: 'html', css: 'css', json: 'json', bash: 'sh', text: 'txt' }[a.lang] || 'txt';
    const blob = new Blob([a.code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = `artifact.${ext}`;
    document.body.appendChild(link); link.click(); link.remove();
    URL.revokeObjectURL(url);
  }));
  overlay.querySelector('#art-close').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}
const artifactsTabBtn = document.getElementById('artifacts-tab-btn');
if (artifactsTabBtn) artifactsTabBtn.addEventListener('click', () => requireVipModel(() => {
  showArtifactsModal();
}));

// ─── Archived view toggle ─────────────────────────────────────────────────────
const archivedToggleBtn = document.getElementById('archived-toggle-btn');
if (archivedToggleBtn) archivedToggleBtn.addEventListener('click', () => requirePassword(() => {
  showingArchived = !showingArchived;
  archivedToggleBtn.textContent = showingArchived ? '💬 Active Chats' : '🗄 Archived';
  archivedToggleBtn.style.color = showingArchived ? 'var(--accent)' : '';
  archivedToggleBtn.style.borderColor = showingArchived ? 'var(--accent)' : 'var(--border)';
  loadConversationList();
}));

// ─── Message bookmarks (stored per-conversation in localStorage) ──────────────
function getBookmarks() {
  try { return JSON.parse(localStorage.getItem('mythic_bookmarks') || '{}'); } catch { return {}; }
}
function saveBookmarks(b) { localStorage.setItem('mythic_bookmarks', JSON.stringify(b)); }
function toggleBookmark(convId, msgIndex, text) {
  if (!convId) return false;
  const all = getBookmarks();
  const list = all[convId] = all[convId] || [];
  const existingIdx = list.findIndex(b => b.msgIndex === msgIndex);
  let nowBookmarked;
  if (existingIdx >= 0) { list.splice(existingIdx, 1); nowBookmarked = false; }
  else { list.push({ msgIndex, text: (text || '').slice(0, 200), ts: Date.now() }); nowBookmarked = true; }
  if (!list.length) delete all[convId];
  saveBookmarks(all);
  return nowBookmarked;
}
function isBookmarked(convId, msgIndex) {
  const list = getBookmarks()[convId] || [];
  return list.some(b => b.msgIndex === msgIndex);
}

const bookmarksBtn = document.getElementById('bookmarks-btn');
function showBookmarksModal() {
  const all = getBookmarks();
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;';
  let rows = '';
  let count = 0;
  Object.entries(all).forEach(([convId, list]) => {
    list.forEach(b => {
      count++;
      rows += `<div class="bm-row" data-conv="${convId}" style="padding:10px;border-bottom:1px solid var(--border);cursor:pointer;font-size:12.5px;color:var(--text);">${b.text.replace(/</g,'&lt;')}</div>`;
    });
  });
  if (!count) rows = '<div style="padding:20px;text-align:center;color:var(--muted);font-size:13px;">No bookmarked messages yet. Use the 🔖 icon on any message.</div>';
  overlay.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px;width:90%;max-width:420px;max-height:70vh;overflow-y:auto;">
    <h3 style="margin:0 0 12px;font-size:16px;">⭐ Bookmarked Messages</h3>
    <div>${rows}</div>
    <button id="bm-close" style="margin-top:14px;width:100%;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:9px;cursor:pointer;font-family:inherit;">Close</button>
  </div>`;
  document.body.appendChild(overlay);
  overlay.querySelectorAll('.bm-row').forEach(row => {
    row.addEventListener('click', () => { openConversation(row.dataset.conv); overlay.remove(); });
  });
  overlay.querySelector('#bm-close').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}
if (bookmarksBtn) bookmarksBtn.addEventListener('click', () => requirePassword(showBookmarksModal));

// ─── Chat statistics ────────────────────────────────────────────────────────
const statsBtn = document.getElementById('stats-btn');
async function showStatsModal() {
  const convs = await fetch('/api/conversations').then(r => r.json()).then(d => d.conversations || []).catch(() => []);
  const streak = await fetch('/api/streak').then(r => r.json()).then(d => d.streak || 0).catch(() => 0);
  let totalMsgsGuess = 0;
  if (activeConvId) {
    const d = await fetch('/api/conversations/' + activeConvId).then(r => r.json()).catch(() => null);
    if (d && d.messages) totalMsgsGuess = d.messages.length;
  }
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;';
  overlay.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px;width:90%;max-width:380px;">
    <h3 style="margin:0 0 14px;font-size:16px;">📊 Chat Statistics</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
      <div style="background:var(--bg);border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:22px;font-weight:700;color:var(--accent);">${convs.length}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px;">Total Chats</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:22px;font-weight:700;color:var(--accent);">🔥 ${streak}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px;">Day Streak</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:22px;font-weight:700;color:var(--accent);">${convs.filter(c=>c.pinned).length}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px;">Pinned</div>
      </div>
      <div style="background:var(--bg);border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:22px;font-weight:700;color:var(--accent);">${totalMsgsGuess}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px;">Messages (this chat)</div>
      </div>
    </div>
    <button id="stats-close" style="margin-top:16px;width:100%;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:9px;cursor:pointer;font-family:inherit;">Close</button>
  </div>`;
  document.body.appendChild(overlay);
  overlay.querySelector('#stats-close').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}
if (statsBtn) statsBtn.addEventListener('click', () => requirePassword(showStatsModal));

// ─── Bookmark button on AI/user messages ───────────────────────────────────
let _msgIndexCounter = 0;
const _origBuildActions2 = buildMsgActions;
buildMsgActions = function(row, textNode, role) {
  const actions = _origBuildActions2(row, textNode, role);
  const myIndex = _msgIndexCounter++;
  row.dataset.msgIndex = myIndex;
  const bmBtn = document.createElement('button');
  bmBtn.type = 'button'; bmBtn.title = 'Bookmark';
  bmBtn.textContent = isBookmarked(activeConvId, myIndex) ? '⭐' : '☆';
  bmBtn.addEventListener('click', () => {
    const on = toggleBookmark(activeConvId, myIndex, textNode.textContent || textNode.innerText || '');
    bmBtn.textContent = on ? '⭐' : '☆';
  });
  actions.appendChild(bmBtn);
  return actions;
};

// ─── Command palette (Ctrl+K) ──────────────────────────────────────────────
function showCommandPalette() {
  const existing = document.getElementById('cmd-palette-overlay');
  if (existing) { existing.remove(); }
  const overlay = document.createElement('div');
  overlay.id = 'cmd-palette-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:600;display:flex;align-items:flex-start;justify-content:center;padding-top:12vh;';
  overlay.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:12px;width:92%;max-width:480px;box-shadow:0 10px 50px rgba(0,0,0,.4);overflow:hidden;">
    <input id="cmd-input" placeholder="Type a command or search chats..." autocomplete="off"
      style="width:100%;box-sizing:border-box;padding:14px 16px;background:transparent;border:none;border-bottom:1px solid var(--border);color:var(--text);font-size:14px;outline:none;font-family:inherit;">
    <div id="cmd-results" style="max-height:320px;overflow-y:auto;"></div>
  </div>`;
  document.body.appendChild(overlay);
  const input = overlay.querySelector('#cmd-input');
  const results = overlay.querySelector('#cmd-results');
  input.focus();

  const staticCommands = [
    { label: '+ New chat', action: () => { startNewChat(); } },
    { label: '⚙ Open Settings', action: () => { settingsModalOverlay.style.display = 'flex'; } },
    { label: '📊 Chat Statistics', action: showStatsModal },
    { label: '⭐ Bookmarked Messages', action: showBookmarksModal },
    { label: '🗄 Toggle Archived View', action: () => archivedToggleBtn && archivedToggleBtn.click() },
    { label: '⬇ Export current chat', action: () => exportBtn.click() },
    { label: '☰ Toggle sidebar', action: () => sidebarToggle.click() },
  ];

  async function renderResults(query) {
    results.innerHTML = '';
    const q = query.trim().toLowerCase();
    const cmdMatches = staticCommands.filter(c => !q || c.label.toLowerCase().includes(q));
    cmdMatches.forEach(c => {
      const row = document.createElement('div');
      row.textContent = c.label;
      row.style.cssText = 'padding:10px 16px;cursor:pointer;font-size:13.5px;color:var(--text);';
      row.addEventListener('mouseenter', () => row.style.background = 'var(--accent-dim)');
      row.addEventListener('mouseleave', () => row.style.background = '');
      row.addEventListener('click', () => { c.action(); overlay.remove(); });
      results.appendChild(row);
    });
    if (q) {
      const convs = await fetch('/api/conversations').then(r => r.json()).then(d => d.conversations || []).catch(() => []);
      const chatMatches = convs.filter(c => c.title.toLowerCase().includes(q)).slice(0, 8);
      chatMatches.forEach(c => {
        const row = document.createElement('div');
        row.textContent = '💬 ' + c.title;
        row.style.cssText = 'padding:10px 16px;cursor:pointer;font-size:13.5px;color:var(--muted);';
        row.addEventListener('mouseenter', () => row.style.background = 'var(--accent-dim)');
        row.addEventListener('mouseleave', () => row.style.background = '');
        row.addEventListener('click', () => { openConversation(c.id); overlay.remove(); });
        results.appendChild(row);
      });
    }
    if (!results.children.length) {
      results.innerHTML = '<div style="padding:16px;text-align:center;color:var(--muted);font-size:12.5px;">No matches.</div>';
    }
  }
  renderResults('');
  input.addEventListener('input', () => renderResults(input.value));
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  input.addEventListener('keydown', e => { if (e.key === 'Escape') overlay.remove(); });
}

// ─── Extra keyboard shortcuts ───────────────────────────────────────────────
document.addEventListener('keydown', e => {
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); showCommandPalette(); return; }
  if (mod && e.shiftKey && e.key.toLowerCase() === 'o') { e.preventDefault(); startNewChat(); return; }
  if (mod && e.key.toLowerCase() === 'b') { e.preventDefault(); sidebarToggle.click(); return; }
  // "/" focuses the message input, unless already typing somewhere
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
    e.preventDefault(); input.focus();
  }
});

// ─── PIN lock (client-side; hashed PIN kept in localStorage) ───────────────
async function _sha256Hex(text) {
  const enc = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest('SHA-256', enc);
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
}

function showPinSetupModal() {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:700;display:flex;align-items:center;justify-content:center;';
  overlay.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:22px;width:90%;max-width:320px;text-align:center;">
    <div style="font-size:30px;margin-bottom:8px;">🔒</div>
    <div style="font-weight:700;font-size:15px;margin-bottom:12px;">Set a 4-digit PIN</div>
    <input id="pin-setup-input" type="password" inputmode="numeric" maxlength="4" placeholder="••••"
      style="width:100%;box-sizing:border-box;text-align:center;letter-spacing:8px;font-size:22px;padding:10px;border-radius:8px;border:1.5px solid var(--border);background:var(--bg);color:var(--text);outline:none;margin-bottom:12px;">
    <div style="display:flex;gap:8px;">
      <button id="pin-setup-save" style="flex:1;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;cursor:pointer;font-family:inherit;">Save</button>
      <button id="pin-setup-cancel" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:10px;cursor:pointer;font-family:inherit;">Cancel</button>
    </div>
  </div>`;
  document.body.appendChild(overlay);
  const pinInput = overlay.querySelector('#pin-setup-input');
  pinInput.focus();
  overlay.querySelector('#pin-setup-cancel').addEventListener('click', () => overlay.remove());
  overlay.querySelector('#pin-setup-save').addEventListener('click', async () => {
    const v = pinInput.value.trim();
    if (!/^\d{4}$/.test(v)) { pinInput.style.borderColor = '#ef4444'; return; }
    const hash = await _sha256Hex(v);
    localStorage.setItem('mythic_pin_hash', hash);
    overlay.remove();
  });
}

function showPinLockScreen() {
  const overlay = document.createElement('div');
  overlay.id = 'pin-lock-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:var(--bg);z-index:9000;display:flex;flex-direction:column;align-items:center;justify-content:center;';
  overlay.innerHTML = `<div style="text-align:center;">
    <div style="font-size:40px;margin-bottom:10px;">🔒</div>
    <div style="font-weight:700;font-size:16px;color:var(--accent);margin-bottom:16px;">Mythic AI Locked</div>
    <input id="pin-unlock-input" type="password" inputmode="numeric" maxlength="4" placeholder="••••"
      style="width:180px;box-sizing:border-box;text-align:center;letter-spacing:8px;font-size:24px;padding:10px;border-radius:8px;border:1.5px solid var(--border);background:var(--panel);color:var(--text);outline:none;">
    <div id="pin-unlock-err" style="color:#ef4444;font-size:12px;margin-top:8px;display:none;">Wrong PIN</div>
  </div>`;
  document.body.appendChild(overlay);
  const pinInput = overlay.querySelector('#pin-unlock-input');
  const errEl = overlay.querySelector('#pin-unlock-err');
  pinInput.focus();
  pinInput.addEventListener('input', async () => {
    if (pinInput.value.length !== 4) return;
    const hash = await _sha256Hex(pinInput.value);
    if (hash === localStorage.getItem('mythic_pin_hash')) {
      overlay.remove();
    } else {
      errEl.style.display = 'block';
      pinInput.value = '';
    }
  });
}

if (localStorage.getItem('mythic_pin_hash')) {
  showPinLockScreen();
}

// Wire up PIN lock enable/disable from Settings (adds a row dynamically so
// it doesn't require restructuring the existing settings modal markup).
(function addPinLockSettingsRow() {
  const settingsModal = document.getElementById('settings-modal');
  const closeBtn = document.getElementById('settings-close-btn');
  if (!settingsModal || !closeBtn) return;
  const section = document.createElement('div');
  section.className = 'settings-section';
  section.style.cssText = 'border-top:1px solid var(--border);padding-top:14px;margin-top:4px;';
  section.innerHTML = `<label style="display:flex;align-items:center;justify-content:space-between;cursor:pointer;">
    <span>🔒 PIN Lock</span>
    <button id="pin-lock-toggle-btn" type="button"
      style="background:none;border:1.5px solid var(--border);color:var(--muted);border-radius:20px;padding:6px 14px;font-size:12px;cursor:pointer;font-family:inherit;">
      ${localStorage.getItem('mythic_pin_hash') ? 'Disable' : 'Enable'}
    </button>
  </label>
  <div class="hint">Locks the app behind a 4-digit PIN on this device. Stored only as a hash in your browser.</div>`;
  closeBtn.parentNode.insertBefore(section, closeBtn);
  section.querySelector('#pin-lock-toggle-btn').addEventListener('click', (e) => {
    if (localStorage.getItem('mythic_pin_hash')) {
      localStorage.removeItem('mythic_pin_hash');
      e.target.textContent = 'Enable';
    } else {
      showPinSetupModal();
      setTimeout(() => { e.target.textContent = localStorage.getItem('mythic_pin_hash') ? 'Disable' : 'Enable'; }, 500);
    }
  });
})();

// ─── Auto-generate a smart AI title after the first exchange in a new chat ──
const _origStreamReply = streamReply;
streamReply = async function(opts) {
  const wasNewChat = !activeConvId;
  await _origStreamReply(opts);
  if (wasNewChat && activeConvId && !(opts && opts.regenerate)) {
    fetch('/api/conversations/' + activeConvId + '/generate-title', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(getUserApiKeys())
    }).then(() => loadConversationList()).catch(() => {});
  }
};
</script>
</body>
</html>
"""

@app.route("/sw.js")
def service_worker_js():
    """Real service worker served from a proper URL so push events work.
    Blob-URL service workers cannot receive push events — this route is
    required for Web Push to function."""
    sw = r"""
const CACHE = 'mythic-ai-v3';

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(['/'])));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  if (e.request.url.includes('/api/')) return;
  e.respondWith(
    fetch(e.request)
      .then(resp => {
        const clone = resp.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
        return resp;
      })
      .catch(() => caches.match(e.request))
  );
});

self.addEventListener('push', e => {
  let data = { title: 'Mythic AI', body: 'You have a new message', icon: '/icon.png', url: '/' };
  try {
    if (e.data) {
      const parsed = e.data.json();
      data = { ...data, ...parsed };
    }
  } catch {}

  e.waitUntil(
    self.registration.showNotification(data.title, {
      body:    data.body,
      icon:    data.icon,
      badge:   '/icon.png',
      tag:     'mythic-ai-reply',
      renotify: true,
      vibrate: [200, 100, 200],
      data:    { url: data.url || '/' },
      actions: [
        { action: 'open',    title: '💬 Open Chat' },
        { action: 'dismiss', title: '✕ Dismiss'   },
      ],
    })
  );
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  if (e.action === 'dismiss') return;
  const target = (e.notification.data && e.notification.data.url) || '/';
  e.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(list => {
      for (const c of list) {
        if (c.url.includes(self.location.origin) && 'focus' in c) {
          c.navigate(target);
          return c.focus();
        }
      }
      if (clients.openWindow) return clients.openWindow(target);
    })
  );
});
"""
    return Response(sw, mimetype="application/javascript",
                    headers={"Service-Worker-Allowed": "/"})


@app.route("/health")
def health_check():
    """Lightweight endpoint for uptime pingers (UptimeRobot, cron-job.org,
    etc.) to hit every 10-14 minutes, keeping a Render free-tier instance
    from spinning down. Does no real work — just confirms the process is alive."""
    return jsonify({"status": "ok", "time": time.time()})


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
        "scope": "/",
        "lang": "en",
        "categories": ["productivity", "utilities"],
        "icons": [
            {"src": "/icon.png",     "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
        "shortcuts": [
            {"name": "New Chat", "url": "/", "description": "Start a new chat"},
        ],
    }
    return Response(
        json.dumps(manifest),
        mimetype="application/manifest+json",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _make_mythic_icon_png(size=192):
    """Generate a real PNG icon for Mythic AI programmatically using only stdlib.
    Draws the teal rounded-rect background + white M-shape — no Pillow needed."""
    import struct, zlib

    W = H = size
    img = bytearray(W * H * 4)

    def set_pixel(x, y, r, g, b, a=255):
        if 0 <= x < W and 0 <= y < H:
            i = (y * W + x) * 4
            img[i], img[i+1], img[i+2], img[i+3] = r, g, b, a

    def fill_rect(x0, y0, x1, y1, r, g, b, a=255):
        for y in range(max(0,y0), min(H,y1)):
            for x in range(max(0,x0), min(W,x1)):
                set_pixel(x, y, r, g, b, a)

    def circle_aa(cx, cy, radius, r, g, b):
        for y in range(cy-radius-1, cy+radius+2):
            for x in range(cx-radius-1, cx+radius+2):
                d = ((x-cx)**2 + (y-cy)**2)**0.5
                alpha = max(0, min(255, int((radius+0.5-d)*255)))
                if alpha > 0 and 0 <= x < W and 0 <= y < H:
                    i = (y*W+x)*4
                    existing_a = img[i+3]
                    blend = alpha / 255
                    img[i]   = int(img[i]   * (1-blend) + r * blend)
                    img[i+1] = int(img[i+1] * (1-blend) + g * blend)
                    img[i+2] = int(img[i+2] * (1-blend) + b * blend)
                    img[i+3] = min(255, existing_a + alpha)

    cr = size // 4
    cx, cy = W // 2, H // 2
    fill_rect(cr, 0, W-cr, H, 16, 163, 127)
    fill_rect(0, cr, W, H-cr, 16, 163, 127)
    circle_aa(cr,   cr,   cr, 16, 163, 127)
    circle_aa(W-cr, cr,   cr, 16, 163, 127)
    circle_aa(cr,   H-cr, cr, 16, 163, 127)
    circle_aa(W-cr, H-cr, cr, 16, 163, 127)

    s = size / 40
    pts = [
        (int(10*s), int(28*s)),
        (int(10*s), int(12*s)),
        (int(20*s), int(22*s)),
        (int(30*s), int(12*s)),
        (int(30*s), int(28*s)),
    ]
    lw = max(2, size // 14)

    def draw_line(x0, y0, x1, y1):
        dx, dy = x1-x0, y1-y0
        steps = max(abs(dx), abs(dy), 1)
        for i in range(steps+1):
            x = int(x0 + dx*i/steps)
            y = int(y0 + dy*i/steps)
            for ox in range(-lw//2, lw//2+1):
                for oy in range(-lw//2, lw//2+1):
                    set_pixel(x+ox, y+oy, 255, 255, 255)

    for i in range(len(pts)-1):
        draw_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

    def png_chunk(name, data):
        crc = zlib.crc32(name + data) & 0xffffffff
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', crc)

    raw_rows = b''
    for y in range(H):
        raw_rows += b'\x00'
        raw_rows += bytes(img[y*W*4:(y+1)*W*4])

    ihdr = struct.pack('>II', W, H) + bytes([8, 6, 0, 0, 0])
    compressed = zlib.compress(raw_rows, 9)

    png  = b'\x89PNG\r\n\x1a\n'
    png += png_chunk(b'IHDR', ihdr)
    png += png_chunk(b'IDAT', compressed)
    png += png_chunk(b'IEND', b'')
    return png


_ICON_192_PNG = None
_ICON_512_PNG = None

def _get_icon(size):
    global _ICON_192_PNG, _ICON_512_PNG
    if size == 192:
        if _ICON_192_PNG is None:
            _ICON_192_PNG = _make_mythic_icon_png(192)
        return _ICON_192_PNG
    else:
        if _ICON_512_PNG is None:
            _ICON_512_PNG = _make_mythic_icon_png(512)
        return _ICON_512_PNG


@app.route("/icon.png")
def pwa_icon_192():
    return Response(_get_icon(192), mimetype="image/png",
                    headers={"Cache-Control": "public, max-age=604800"})

@app.route("/icon-512.png")
def pwa_icon_512():
    return Response(_get_icon(512), mimetype="image/png",
                    headers={"Cache-Control": "public, max-age=604800"})

@app.route("/favicon.ico")
def favicon():
    return Response(_get_icon(192), mimetype="image/png",
                    headers={"Cache-Control": "public, max-age=604800"})


@app.route("/")
@login_required
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")


@app.route("/api/conversations", methods=["GET"])
@login_required
def api_list_conversations():
    show_archived = request.args.get("archived", "0") == "1"
    convs = list_conversations(current_username())
    if show_archived:
        convs = [c for c in convs if c.get("archived")]
    else:
        convs = [c for c in convs if not c.get("archived")]
    return jsonify({"conversations": convs})


@app.route("/api/folders", methods=["GET"])
@login_required
def api_list_folders():
    return jsonify({"folders": list_folders(current_username())})


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
    """Updates one or more of: title, folder, pinned, archived. At least one
    field must be present; unspecified fields are left unchanged."""
    data = request.get_json(force=True) or {}
    username = current_username()
    conv = load_conversation(username, conv_id)
    if conv is None:
        return jsonify({"error": "not found"}), 404

    changed = {}
    if "title" in data:
        new_title = (data.get("title") or "").strip()[:120]
        if not new_title:
            return jsonify({"error": "title cannot be empty"}), 400
        conv["title"] = new_title
        conv["title_is_custom"] = True
        changed["title"] = new_title
    if "folder" in data:
        folder = (data.get("folder") or "").strip()[:60] or None
        conv["folder"] = folder
        changed["folder"] = folder
    if "pinned" in data:
        conv["pinned"] = bool(data.get("pinned"))
        changed["pinned"] = conv["pinned"]
    if "archived" in data:
        conv["archived"] = bool(data.get("archived"))
        changed["archived"] = conv["archived"]

    if not changed:
        return jsonify({"error": "no recognized fields to update "
                                  "(expected title/folder/pinned/archived)"}), 400

    save_conversation(username, conv_id, conv)
    return jsonify({"status": "updated", **changed})


@app.route("/api/conversations/<conv_id>/duplicate", methods=["POST"])
@login_required
def api_duplicate_conversation(conv_id):
    username = current_username()
    conv = load_conversation(username, conv_id)
    if conv is None:
        return jsonify({"error": "not found"}), 404
    new_id = str(uuid.uuid4())
    new_conv = {
        "title": (conv.get("title") or "New chat") + " (copy)",
        "messages": json.loads(json.dumps(conv.get("messages", []))),  # deep copy
        "folder": conv.get("folder"),
        "pinned": False,
        "archived": False,
    }
    save_conversation(username, new_id, new_conv)
    return jsonify({"status": "duplicated", "id": new_id, "title": new_conv["title"]})


def _collect_full_reply(chunks):
    return "".join(chunks)


@app.route("/api/conversations/<conv_id>/generate-title", methods=["POST"])
@login_required
def api_generate_title(conv_id):
    """Asks the AI to write a short, punchy title from the first exchange in
    the conversation, replacing the naive first-40-characters title. Safe to
    call any time; falls back to leaving the title unchanged if the AI can't
    be reached."""
    username = current_username()
    conv = load_conversation(username, conv_id)
    if conv is None:
        return jsonify({"error": "not found"}), 404
    messages = conv.get("messages", [])
    if not messages:
        return jsonify({"error": "conversation has no messages yet"}), 400

    convo_excerpt = []
    for m in messages[:4]:
        text = "".join(p.get("text", "") for p in m.get("parts", []) if "text" in p)
        if text:
            speaker = "User" if m["role"] == "user" else "Assistant"
            convo_excerpt.append(f"{speaker}: {text[:300]}")
    excerpt = "\n".join(convo_excerpt)
    if not excerpt.strip():
        return jsonify({"error": "no text content to summarize"}), 400

    title_prompt = [{"role": "user", "parts": [{"text":
        "Write a short chat title (max 6 words, no quotes, no trailing "
        "punctuation, plain text only) that summarizes this conversation:\n\n"
        f"{excerpt}"
    }]}]
    data = request.get_json(silent=True) or {}
    user_groq_key = (data.get("groq_api_key") or "").strip()
    user_cerebras_key = (data.get("cerebras_api_key") or "").strip()

    try:
        raw_title = _collect_full_reply(
            auto_stream_chunks(None, title_prompt, SYSTEM_PROMPT, user_groq_key, user_cerebras_key)
        ).strip()
    except Exception:
        raw_title = ""

    raw_title = raw_title.strip().strip('"').strip("'")
    raw_title = re.sub(r'^\[Instructions:.*?\]\s*', '', raw_title, flags=re.DOTALL)
    # Take just the first line/sentence in case the model added extra
    # commentary despite instructions — truncate rather than reject outright,
    # so a slightly-verbose reply still produces a usable title.
    raw_title = raw_title.split("\n")[0].strip()
    if not raw_title:
        return jsonify({"status": "unchanged", "title": conv.get("title", "New chat")})

    new_title = raw_title[:60]
    conv["title"] = new_title
    save_conversation(username, conv_id, conv)
    return jsonify({"status": "generated", "title": new_title})


# ── Lightweight in-memory response cache ─────────────────────────────────────
# Used for deterministic, repeat-prone calls (follow-up suggestions, chat-title
# generation) where an identical prompt is likely to recur and re-hitting the
# API adds latency/quota use for no benefit. NOT used for normal chat turns,
# since conversation context differs on every message. TTL + size-capped so it
# never grows unbounded; resets on process restart (no external cache needed).
_response_cache = {}
_RESPONSE_CACHE_TTL_SECONDS = 600
_RESPONSE_CACHE_MAX_ENTRIES = 300


def _cache_key(label, model, messages):
    raw = json.dumps(messages, sort_keys=True) + "|" + label + "|" + (model or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key):
    entry = _response_cache.get(key)
    if not entry:
        return None
    text, ts = entry
    if time.time() - ts > _RESPONSE_CACHE_TTL_SECONDS:
        _response_cache.pop(key, None)
        return None
    return text


def _cache_set(key, text):
    if len(_response_cache) > _RESPONSE_CACHE_MAX_ENTRIES:
        _response_cache.pop(next(iter(_response_cache)), None)
    _response_cache[key] = (text, time.time())


def to_openai_messages(gemini_messages, system_prompt):
    """Convert stored Gemini-format messages to OpenAI-compatible chat format.
    Used by both Groq and Cerebras (both use the same OpenAI-style chat API)."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in gemini_messages:
        role = "user" if m["role"] == "user" else "assistant"
        text = "".join(p.get("text", "") for p in m["parts"] if "text" in p)
        msgs.append({"role": role, "content": text})
    return msgs


def _openai_style_stream(url, api_key, model, messages, provider_label):
    """Shared streaming logic for Groq/Cerebras (both are OpenAI-compatible).
    Yields nothing at all on ANY failure (auth, rate limit, timeout, invalid
    model, network error, 4xx/5xx) so the caller can silently fall through to
    the next provider without ever exposing a provider error to the user.

    On Vercel/serverless, streaming responses are buffered and can be cut off,
    so we fall back to a single non-streaming request that returns the full
    reply in one go — the frontend still displays it, just not word-by-word."""
    if not api_key:
        print(f"[{provider_label}] skipped: no API key configured")
        return

    # Non-streaming path on serverless — avoids Vercel's edge buffer cutting
    # the response short mid-stream.
    if IS_SERVERLESS:
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "stream": False, "max_tokens": 2048},
                timeout=45,
            )
        except requests.RequestException as e:
            print(f"[{provider_label}] network error: {e}")
            return
        if resp.status_code != 200:
            try: body_preview = resp.text[:500]
            except Exception: body_preview = "<unreadable>"
            print(f"[{provider_label}] HTTP {resp.status_code}: {body_preview}")
            return
        try:
            obj = resp.json()
            content = obj["choices"][0]["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"[{provider_label}] bad JSON: {e}")
            return
        # Yield the full reply in reasonable chunks so the frontend still
        # renders progressively even though the network delivered it all at once
        step = 80
        for i in range(0, len(content), step):
            yield content[i:i + step]
        return

    # Streaming path — the normal, preferred flow on always-on hosts
    max_retries = 2
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "stream": True, "max_tokens": 2048},
                stream=True, timeout=60,
            )
        except requests.RequestException as e:
            print(f"[{provider_label}] network error: {e}")
            return
        if resp.status_code == 429 and attempt < max_retries:
            # Rate-limited — brief backoff, then retry the same model/provider
            # before giving up on it entirely.
            print(f"[{provider_label}] rate-limited (429), retrying in "
                  f"{1.5 * (attempt + 1):.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(1.5 * (attempt + 1))
            continue
        break

    if resp.status_code != 200:
        try:
            body_preview = resp.text[:500]
        except Exception:
            body_preview = "<unreadable>"
        print(f"[{provider_label}] HTTP {resp.status_code}: {body_preview}")
        return

    for raw_line in resp.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not line.startswith("data:"):
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


def _stream_with_model_fallback(url, api_key, primary_model, fallback_model, messages, label):
    """Tries `primary_model` first (e.g. a model the person picked in the
    Model Manager). If it yields nothing at all — invalid/decommissioned
    model, 404, etc. — automatically retries once on `fallback_model` (the
    server's configured default) before giving up on this provider, so a
    stale or unavailable model choice doesn't just silently fail."""
    models_to_try = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models_to_try.append(fallback_model)
    for m in models_to_try:
        got_any = False
        for chunk in _openai_style_stream(url, api_key, m, messages, f"{label}:{m}"):
            got_any = True
            yield chunk
        if got_any:
            return


def groq_stream_chunks(messages, api_key=None, model=None):
    """Stream from Groq (primary chat provider). Uses a person's own key
    (from Settings) when provided, otherwise the server's GROQ_API_KEY. An
    optional `model` overrides GROQ_MODEL for this request (falls back to
    GROQ_MODEL automatically if the override is unavailable)."""
    yield from _stream_with_model_fallback(
        "https://api.groq.com/openai/v1/chat/completions",
        api_key or GROQ_API_KEY, model or GROQ_MODEL, GROQ_MODEL, messages, "Groq",
    )


def cerebras_stream_chunks(messages, api_key=None, model=None):
    """Stream from Cerebras (automatic fallback if Groq is unavailable). Uses
    a person's own key (from Settings) when provided, otherwise the
    server's CEREBRAS_API_KEY. An optional `model` overrides CEREBRAS_MODEL
    for this request (falls back to CEREBRAS_MODEL automatically if the
    override is unavailable)."""
    yield from _stream_with_model_fallback(
        "https://api.cerebras.ai/v1/chat/completions",
        api_key or CEREBRAS_API_KEY, model or CEREBRAS_MODEL, CEREBRAS_MODEL, messages, "Cerebras",
    )


def _quick_completion(messages, api_key_groq=None, api_key_cerebras=None, max_tokens=20):
    """Non-streaming, short completion used for auxiliary tasks like AI title
    generation — tries Groq then Cerebras (same silent-fallback pattern as
    chat), returns plain text or None if both fail. Kept deliberately small
    (max_tokens) since this is just for a 3-6 word chat title, not a real
    reply, so it stays fast and cheap."""
    groq_key = (api_key_groq or "").strip() or GROQ_API_KEY
    cerebras_key = (api_key_cerebras or "").strip() or CEREBRAS_API_KEY

    for key, model, url, label in (
        (groq_key, GROQ_MODEL, "https://api.groq.com/openai/v1/chat/completions", "Groq"),
        (cerebras_key, CEREBRAS_MODEL, "https://api.cerebras.ai/v1/chat/completions", "Cerebras"),
    ):
        if not key:
            continue
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "stream": False, "max_tokens": max_tokens, "temperature": 0.4},
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            obj = resp.json()
            content = (obj["choices"][0]["message"]["content"] or "").strip()
            if content:
                return content
        except Exception as e:
            print(f"[TitleGen/{label}] failed: {e}")
            continue
    return None


def generate_smart_title(first_user_message, first_ai_reply, api_key_groq=None, api_key_cerebras=None):
    """Asks the AI for a short, natural chat title based on the first
    exchange — the same pattern ChatGPT/Claude use, instead of just
    truncating the raw first message. Falls back to make_title() if the
    AI call fails for any reason, so a title is always produced."""
    user_msg = (first_user_message or "").strip()
    user_msg = re.sub(r'^\[Instructions:.*?\]\s*', '', user_msg, flags=re.DOTALL)
    ai_reply = (first_ai_reply or "").strip()

    if not user_msg and not ai_reply:
        return "New chat"

    prompt = (
        "Generate a short, natural chat title (3-6 words, no quotes, no punctuation "
        "at the end, no emoji, title case) that summarizes what this conversation is "
        "about. Reply with ONLY the title text, nothing else.\n\n"
        f"User: {user_msg[:400]}\n"
        f"Assistant: {ai_reply[:400]}"
    )
    messages = [
        {"role": "system", "content": "You generate concise chat titles. Reply with only the title, no extra text."},
        {"role": "user", "content": prompt},
    ]
    result = _quick_completion(messages, api_key_groq, api_key_cerebras, max_tokens=16)
    if result:
        title = result.strip().strip('"').strip("'").split("\n")[0].strip()
        title = re.sub(r'[.!?]+$', '', title).strip()
        if title:
            return title[:60]
    return make_title(first_user_message)


def auto_stream_chunks(gemini_payload, gemini_messages, system_prompt=None,
                        user_groq_key=None, user_cerebras_key=None,
                        groq_model=None, cerebras_model=None):
    """Groq first, Cerebras as a silent automatic fallback.
    Never asks the user to pick a provider and never exposes provider errors —
    if Groq yields nothing (rate limit, timeout, invalid model, network error,
    4xx/5xx), we just move on to Cerebras with no visible interruption.
    If the person supplied their own API key(s) in Settings, those are tried
    first (and exclusively, in that provider's slot) before the server's key.
    `groq_model`/`cerebras_model` optionally override the chosen model within
    each provider (see Model Manager) — invalid choices auto-fall-back to the
    server's default model for that provider before moving to the next provider."""
    sp = system_prompt or SYSTEM_PROMPT
    openai_msgs = to_openai_messages(gemini_messages, sp)

    order = []
    groq_key = (user_groq_key or "").strip() or GROQ_API_KEY
    cerebras_key = (user_cerebras_key or "").strip() or CEREBRAS_API_KEY
    if PROVIDER in ("auto", "groq") and groq_key:
        order.append(("Groq", lambda: groq_stream_chunks(openai_msgs, groq_key, groq_model)))
    if PROVIDER in ("auto", "cerebras") and cerebras_key:
        order.append(("Cerebras", lambda: cerebras_stream_chunks(openai_msgs, cerebras_key, cerebras_model)))

    if not order:
        # Kept short and free of internal setup instructions — see chat
        # request for context.
        yield "I'm not able to reply right now — please try again shortly."
        return

    for _name, fn in order:
        collected = False
        try:
            for chunk in fn():
                collected = True
                yield chunk
            if collected:
                return
        except Exception as e:
            print(f"[{_name}] unexpected error: {e}")

    # All configured providers failed silently — keep it short and generic.
    yield "I couldn't get a reply just now. Please try again in a moment."


# --- Model selector (cosmetic tiers over the same underlying providers) -----
VIP_PASSWORD = os.environ.get("VIP_PASSWORD", "1254")

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


@app.route("/api/streak", methods=["GET"])
@login_required
def api_streak():
    """Returns the current user's daily chat streak length (0 if they've
    never chatted or their streak has already lapsed)."""
    username = current_username()
    return jsonify({"streak": _get_user_streak(username)})


@app.route("/api/push/vapid-public-key", methods=["GET"])
def push_vapid_key():
    if not VAPID_PUBLIC_KEY:
        return jsonify({"error": "push not configured"}), 503
    # length_ok: a valid P-256 uncompressed-point key, base64url-encoded
    # with no padding, is always exactly 87 characters. If this is false,
    # the env var was pasted with extra/missing characters (whitespace,
    # truncation, wrong key type) and the browser will reject it with
    # "applicationServerKey is not valid".
    return jsonify({
        "publicKey": VAPID_PUBLIC_KEY,
        "length": len(VAPID_PUBLIC_KEY),
        "length_ok": len(VAPID_PUBLIC_KEY) == 87,
    })


@app.route("/api/push/subscribe", methods=["POST"])
@login_required
def push_subscribe():
    """Save (or update) a browser's push subscription, tagged with the
    current (anonymous) username so re-engagement notifications can be
    targeted at this specific person later."""
    data = request.get_json(force=True) or {}
    sub = data.get("subscription")
    if not sub or not sub.get("endpoint"):
        return jsonify({"error": "invalid subscription"}), 400
    sub_id = str(uuid.uuid5(uuid.NAMESPACE_URL, sub["endpoint"]))
    sub = dict(sub)
    sub["_username"] = current_username()
    _save_push_subscription(sub_id, sub)
    return jsonify({"status": "subscribed", "id": sub_id})


@app.route("/api/push/unsubscribe", methods=["POST"])
@login_required
def push_unsubscribe():
    data = request.get_json(force=True) or {}
    endpoint = (data.get("endpoint") or "").strip()
    if endpoint:
        sub_id = str(uuid.uuid5(uuid.NAMESPACE_URL, endpoint))
        _delete_push_subscription(sub_id)
    return jsonify({"status": "unsubscribed"})


@app.route("/api/push/test-reengagement", methods=["POST"])
@login_required
def push_test_reengagement():
    """Manually fire one reengagement pass right now (bypassing the hourly
    wait) so you can verify VAPID/subscriptions/logic are actually working
    without sitting around for an hour. Remove or protect this in production."""
    if not _PUSH_AVAILABLE:
        return jsonify({"error": "pywebpush not installed"}), 500
    if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        return jsonify({"error": "VAPID_PRIVATE_KEY / VAPID_PUBLIC_KEY not set on the server"}), 400
    _load_push_subscriptions()
    usernames = _subscribed_usernames()
    if not usernames:
        return jsonify({"error": "no push subscriptions registered yet — enable notifications in Settings first"}), 400
    _run_reengagement_pass()
    return jsonify({"status": "ran", "subscribed_users": len(usernames)})


@app.route("/api/push/test", methods=["POST"])
@login_required
def push_test():
    send_push_notification(
        title="Mythic AI",
        body="🎉 Push notifications are working!",
        url="/",
    )
    return jsonify({"status": "sent"})


# ── Cron endpoint for scheduled notifications ─────────────────────────────────
# SCHEDULE SUMMARY:
#   - Render (or any always-on host): nothing to configure — the in-process
#     background thread (_reengagement_loop) fires automatically every
#     _REENGAGEMENT_CHECK_INTERVAL_SECONDS (1 hour).
#   - Vercel (serverless): the in-process thread never runs there (functions
#     freeze between requests), so this endpoint must be triggered by an
#     EXTERNAL cron, once a day at 12:00 (noon). Configure it with:
#
#   vercel.json:
#     {
#       "crons": [
#         { "path": "/api/cron/reengagement", "schedule": "0 12 * * *" }
#       ]
#     }
#
#   "0 12 * * *" = once a day at 12:00 in the project's configured cron
#   timezone (UTC by default on Vercel — adjust the hour if you need a
#   specific local noon).
#
# Protection: set CRON_SECRET as an environment variable and pass it in a
# header:  Authorization: Bearer <CRON_SECRET>
#          or in a query string: /api/cron/reengagement?secret=<CRON_SECRET>
# If CRON_SECRET is unset, the endpoint is open (fine for dev only).
CRON_SECRET = os.environ.get("CRON_SECRET", "").strip()

@app.route("/api/cron/reengagement", methods=["GET", "POST"])
def cron_reengagement():
    if CRON_SECRET:
        provided = ""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:].strip()
        elif "x-vercel-cron-secret" in request.headers:
            provided = request.headers.get("x-vercel-cron-secret", "").strip()
        else:
            provided = request.args.get("secret", "").strip()
        if provided != CRON_SECRET:
            return jsonify({"error": "unauthorized"}), 401
    try:
        _run_reengagement_pass()
        return jsonify({"status": "ok", "subscribers": len(_push_subscriptions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/health: quick diagnostic so admins can see what's configured ────────
@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "app": "Mythic AI",
        "serverless": IS_SERVERLESS,
        "providers": {
            "groq":     {"configured": bool(GROQ_API_KEY),     "model": GROQ_MODEL},
            "cerebras": {"configured": bool(CEREBRAS_API_KEY), "model": CEREBRAS_MODEL},
        },
        "image_generation": {
            "nano_banana":  bool(NANO_BANANA_API_KEY),
            "huggingface":  bool(HF_API_KEY),
            "pollinations": True,  # always available, no key needed
        },
        "push_notifications": {
            "configured": _PUSH_AVAILABLE and bool(VAPID_PRIVATE_KEY) and bool(VAPID_PUBLIC_KEY),
            "subscribers": len(_push_subscriptions),
            "cron_endpoint": "/api/cron/reengagement",
            "cron_schedule": "Render: every 1 hour (built-in thread). Vercel: once daily at 12:00 via external cron.",
            "cron_secret_set": bool(CRON_SECRET),
        },
        "hint": ("Add GROQ_API_KEY or CEREBRAS_API_KEY as environment variables "
                 "if 'configured' is false. On Vercel, also set up a cron job "
                 "hitting /api/cron/reengagement once a day at 12:00 for notifications.")
    })


# --- Weather (Open-Meteo — free, no API key needed, works for any country/city) ---
_WMO_ICON = {
    0: ("☀️", "Clear sky"), 1: ("🌤", "Mainly clear"), 2: ("⛅", "Partly cloudy"), 3: ("☁️", "Overcast"),
    45: ("🌫", "Fog"), 48: ("🌫", "Freezing fog"),
    51: ("🌦", "Light drizzle"), 53: ("🌦", "Drizzle"), 55: ("🌧", "Dense drizzle"),
    56: ("🌧", "Freezing drizzle"), 57: ("🌧", "Freezing drizzle"),
    61: ("🌧", "Light rain"), 63: ("🌧", "Rain"), 65: ("🌧", "Heavy rain"),
    66: ("🌧", "Freezing rain"), 67: ("🌧", "Freezing rain"),
    71: ("🌨", "Light snow"), 73: ("🌨", "Snow"), 75: ("❄️", "Heavy snow"), 77: ("❄️", "Snow grains"),
    80: ("🌦", "Rain showers"), 81: ("🌧", "Rain showers"), 82: ("⛈", "Violent rain showers"),
    85: ("🌨", "Snow showers"), 86: ("❄️", "Snow showers"),
    95: ("⛈", "Thunderstorm"), 96: ("⛈", "Thunderstorm with hail"), 99: ("⛈", "Thunderstorm with hail"),
}


def _wmo(code):
    return _WMO_ICON.get(code, ("🌡", "Unknown"))


def _aqi_label(us_aqi):
    if us_aqi is None:
        return None
    if us_aqi <= 50: return f"{us_aqi} (Good)"
    if us_aqi <= 100: return f"{us_aqi} (Moderate)"
    if us_aqi <= 150: return f"{us_aqi} (Unhealthy for sensitive groups)"
    if us_aqi <= 200: return f"{us_aqi} (Unhealthy)"
    if us_aqi <= 300: return f"{us_aqi} (Very unhealthy)"
    return f"{us_aqi} (Hazardous)"


@app.route("/api/weather", methods=["POST"])
@login_required
def api_weather():
    data = request.get_json(force=True) or {}
    location_name = (data.get("location") or "").strip()
    lat = data.get("lat")
    lon = data.get("lon")
    display_name = location_name

    try:
        if lat is None or lon is None:
            if not location_name:
                return jsonify({"error": "Enter a city or use your location."}), 400
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location_name, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            ).json()
            results = geo.get("results") or []
            if not results:
                return jsonify({"error": f'Could not find "{location_name}".'}), 404
            top = results[0]
            lat, lon = top["latitude"], top["longitude"]
            parts = [top.get("name")]
            if top.get("admin1") and top.get("admin1") != top.get("name"):
                parts.append(top["admin1"])
            if top.get("country"):
                parts.append(top["country"])
            display_name = ", ".join(p for p in parts if p)

        fr = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,"
                           "wind_speed_10m,pressure_msl,visibility",
                "hourly": "temperature_2m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,uv_index_max,"
                         "sunrise,sunset",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=10,
        ).json()

        current = fr.get("current", {})
        hourly_raw = fr.get("hourly", {})
        daily_raw = fr.get("daily", {})

        if not display_name:
            try:
                rev = requests.get(
                    "https://geocoding-api.open-meteo.com/v1/reverse",
                    params={"latitude": lat, "longitude": lon, "language": "en", "format": "json"},
                    timeout=8,
                ).json()
                rr = (rev.get("results") or [None])[0]
                if rr:
                    display_name = ", ".join(p for p in [rr.get("name"), rr.get("country")] if p)
            except Exception:
                pass
            display_name = display_name or f"{lat:.2f}, {lon:.2f}"

        icon, condition = _wmo(current.get("weather_code"))

        aqi = None
        try:
            aq = requests.get(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                params={"latitude": lat, "longitude": lon, "current": "us_aqi"},
                timeout=8,
            ).json()
            aqi = _aqi_label((aq.get("current") or {}).get("us_aqi"))
        except Exception:
            pass

        hourly = []
        times = hourly_raw.get("time", [])
        temps = hourly_raw.get("temperature_2m", [])
        codes = hourly_raw.get("weather_code", [])
        now_iso = current.get("time", "")
        start_idx = 0
        for i, t in enumerate(times):
            if t >= now_iso:
                start_idx = i
                break
        for i in range(start_idx, min(start_idx + 8, len(times))):
            hi, _ = _wmo(codes[i] if i < len(codes) else None)
            hour_label = times[i][11:16] if len(times[i]) >= 16 else times[i]
            hourly.append({"time": hour_label, "icon": hi, "temp": round(temps[i]) if i < len(temps) else None})

        daily = []
        d_times = daily_raw.get("time", [])
        d_max = daily_raw.get("temperature_2m_max", [])
        d_min = daily_raw.get("temperature_2m_min", [])
        d_codes = daily_raw.get("weather_code", [])
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, t in enumerate(d_times):
            try:
                y, m, dd = [int(x) for x in t.split("-")]
                import datetime as _dt
                wd = weekday_names[_dt.date(y, m, dd).weekday()]
            except Exception:
                wd = t
            di, _ = _wmo(d_codes[i] if i < len(d_codes) else None)
            daily.append({
                "day": "Today" if i == 0 else wd,
                "icon": di,
                "max": round(d_max[i]) if i < len(d_max) else None,
                "min": round(d_min[i]) if i < len(d_min) else None,
            })

        weather = {
            "location": display_name,
            "icon": icon,
            "condition": condition,
            "temp": round(current.get("temperature_2m", 0)),
            "feels_like": round(current.get("apparent_temperature", 0)),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": round(current.get("wind_speed_10m", 0)),
            "pressure": round(current.get("pressure_msl")) if current.get("pressure_msl") is not None else None,
            "visibility": round((current.get("visibility") or 0) / 1000, 1) if current.get("visibility") is not None else None,
            "uv": (daily_raw.get("uv_index_max") or [None])[0],
            "sunrise": (daily_raw.get("sunrise") or [None])[0][11:16] if (daily_raw.get("sunrise") or [None])[0] else None,
            "sunset": (daily_raw.get("sunset") or [None])[0][11:16] if (daily_raw.get("sunset") or [None])[0] else None,
            "aqi": aqi,
            "hourly": hourly,
            "daily": daily,
        }
        return jsonify({"weather": weather})
    except requests.RequestException as e:
        return jsonify({"error": f"Weather service unavailable: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"Could not process weather data: {e}"}), 500


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    conv_id = data.get("conversation_id")
    attachment = data.get("attachment")  # {name, mimeType, dataBase64} or None
    user_name = (data.get("user_name") or "").strip()[:60]  # what Mythic AI should call the user
    requested_model = (data.get("model") or DEFAULT_MODEL_ID).strip()
    regenerate = bool(data.get("regenerate"))
    ephemeral = bool(data.get("ephemeral"))
    # Optional per-person "bring your own API key" override, set in Settings.
    user_groq_key = (data.get("groq_api_key") or "").strip()
    user_cerebras_key = (data.get("cerebras_api_key") or "").strip()
    # Optional model overrides from the Model Manager (falls back to the
    # server's default model automatically if unavailable — see
    # _stream_with_model_fallback).
    groq_model_override = (data.get("groq_model") or "").strip() or None
    cerebras_model_override = (data.get("cerebras_model") or "").strip() or None
    continue_reply = bool(data.get("continue_reply"))

    if ephemeral:
        if not user_message:
            return jsonify({"error": "message is required"}), 400
        temp_messages = [{"role": "user", "parts": [{"text": user_message}]}]

        def generate_ephemeral():
            for chunk in auto_stream_chunks(None, temp_messages, SYSTEM_PROMPT,
                                             user_groq_key, user_cerebras_key):
                yield chunk.encode("utf-8")

        return Response(stream_with_context(generate_ephemeral()),
                         mimetype="text/plain; charset=utf-8")

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
        conv = {"title": make_title(user_message), "messages": [],
                "folder": None, "pinned": False, "archived": False}

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
            filename = attachment.get("name", "file")
            extracted_text, extract_note = extract_text_from_attachment(filename, mime_type, raw)
            if extracted_text is not None:
                doc_block = f"\n\n[Attached file: {filename}]\n{extracted_text}"
                if extract_note:
                    doc_block += f"\n[Note: {extract_note}]"
                user_parts.append({"text": doc_block})
            elif mime_type.startswith("image/"):
                user_parts.append({"text": (
                    f"\n\n[Attached image: {filename} — I can't see images (only Groq/Cerebras "
                    f"text models are used here, no vision API is configured). If you need help "
                    f"with what's in this image, please describe it in words.]"
                )})
            elif extract_note:
                user_parts.append({"text": f"\n\n[Attached file: {filename} — {extract_note}]"})
            user_parts.append({
                "inline_data": {"mime_type": mime_type, "data": attachment["dataBase64"]}
            })
            attachment_meta = {"name": filename, "mimeType": mime_type}

        user_entry = {"role": "user", "parts": user_parts}
        if attachment_meta:
            user_entry["attachment_meta"] = attachment_meta
        messages.append(user_entry)

        _update_user_activity(username)

    effective_system_prompt = SYSTEM_PROMPT
    requested_mode = (data.get("mode") or "default").strip()
    mode_addition = MODE_PROMPTS.get(requested_mode, "")
    if mode_addition:
        effective_system_prompt += " " + mode_addition
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

    is_first_exchange = (not regenerate) and len(messages) == 1  # just the user message so far

    def generate():
        full_reply = []
        chunk_source = auto_stream_chunks(None, messages, effective_system_prompt,
                                           user_groq_key, user_cerebras_key)

        for chunk in chunk_source:
            full_reply.append(chunk)
            yield chunk.encode("utf-8")
        reply_text = "".join(full_reply)
        messages.append({"role": "model", "parts": [{"text": reply_text}]})

        # AI-generated smart title — only for the very first exchange in a
        # conversation, and only if the person hasn't already renamed the
        # chat themselves (title_is_custom guards against overwriting that).
        if is_first_exchange and not conv.get("title_is_custom"):
            try:
                conv["title"] = generate_smart_title(
                    user_message, reply_text, user_groq_key, user_cerebras_key
                )
            except Exception as e:
                print(f"[SmartTitle] failed, keeping fallback title: {e}")

        save_conversation(username, conv_id, conv)

    resp = Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["X-Conversation-Id"] = conv_id
    return resp


@app.route("/api/temp-image/<img_id>", methods=["GET"])
def serve_temp_image(img_id):
    entry = _TEMP_IMAGES.get(img_id)
    if not entry:
        return jsonify({"error": "not found or expired"}), 404
    return Response(entry["data"], mimetype=entry["mime_type"])


@app.route("/api/generate-file", methods=["POST"])
@login_required
def generate_file():
    """Generates a downloadable file (PDF, Word doc, or plain text) from
    text content — used for "generate a PDF / document / downloadable
    file" requests. Returns base64 file bytes + filename + mime type."""
    try:
        data = request.get_json(force=True) or {}
        content = (data.get("content") or "").strip()
        fmt = (data.get("format") or "pdf").strip().lower()
        title = (data.get("title") or "Mythic AI Document").strip()[:100]
        filename_base = "".join(
            c for c in title if c.isalnum() or c in " -_"
        ).strip().replace(" ", "-") or "Mythic-AI-Document"

        if not content:
            return jsonify({"error": "content is required"}), 400

        if fmt in ("docx", "word", "doc"):
            docx_bytes = generate_docx_bytes(title, content)
            if docx_bytes is not None:
                return jsonify({
                    "file": base64.b64encode(docx_bytes).decode("utf-8"),
                    "filename": f"{filename_base}.docx",
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                })
            return jsonify({
                "file": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
                "filename": f"{filename_base}.txt",
                "mimeType": "text/plain",
                "note": "Word (.docx) generation isn't set up on this server "
                        "(needs `pip install python-docx`) — sent as a plain "
                        "text file instead.",
            })

        if fmt in ("txt", "text"):
            return jsonify({
                "file": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
                "filename": f"{filename_base}.txt",
                "mimeType": "text/plain",
            })

        pdf_bytes = generate_pdf_bytes(title, content)
        return jsonify({
            "file": base64.b64encode(pdf_bytes).decode("utf-8"),
            "filename": f"{filename_base}.pdf",
            "mimeType": "application/pdf",
        })
    except Exception as e:
        return jsonify({"error": f"File generation failed: {e}"}), 500


@app.route("/api/generate-image", methods=["POST"])
@login_required
def generate_image():
    try:
        data = request.get_json(force=True) or {}
        prompt = data.get("prompt", "").strip()
        image_b64 = data.get("imageBase64")
        mime_type = data.get("mimeType", "image/jpeg")
        style = data.get("style", "").strip()
        # Per-request watermark settings (from Settings → Image Watermark in
        # the frontend, or sane defaults if not sent). Every branch below
        # routes its result through _finalize_generated_image() so the mark
        # can never be skipped for any current or future image provider.
        watermark_opts = data.get("watermark") or {}
        if not prompt:
            return jsonify({"error": "prompt required"}), 400

        full_prompt = f"{prompt}, {style}" if style else prompt

        prompt_lower = full_prompt.lower()
        is_book      = any(w in prompt_lower for w in ["book cover", "book", "novel cover", "textbook"])
        is_portrait  = any(w in prompt_lower for w in ["portrait", "person", "face", "selfie", "photo of"])
        is_logo      = any(w in prompt_lower for w in ["logo", "icon", "emblem", "badge"])
        is_anime     = any(w in prompt_lower for w in ["anime", "ghibli", "manga", "cartoon", "illustration"])
        is_landscape = any(w in prompt_lower for w in ["landscape", "scenery", "nature", "city", "skyline", "aerial"])

        quality_tail = ", masterpiece, best quality, highly detailed, sharp focus, professional"

        if is_book:
            enhanced = (
                f"{full_prompt}, professional book cover design, elegant typography layout, "
                f"dramatic lighting, rich colors, visually striking, award-winning cover art, "
                f"publishing industry standard{quality_tail}"
            )
            width, height = 512, 768
        elif is_portrait:
            enhanced = (
                f"{full_prompt}, cinematic portrait photography, soft studio lighting, "
                f"8K ultra-detailed, DSLR quality, photorealistic{quality_tail}"
            )
            width, height = 512, 768
        elif is_logo:
            enhanced = (
                f"{full_prompt}, clean vector style, minimalist, professional brand design, "
                f"flat design, scalable, high contrast{quality_tail}"
            )
            width, height = 768, 768
        elif is_anime:
            enhanced = (
                f"{full_prompt}, vibrant anime art, clean linework, beautiful coloring, "
                f"Studio Ghibli quality, cel shading{quality_tail}"
            )
            width, height = 768, 768
        elif is_landscape:
            enhanced = (
                f"{full_prompt}, epic wide shot, golden hour lighting, ultra-wide, "
                f"breathtaking scenery, National Geographic quality{quality_tail}"
            )
            width, height = 896, 512
        else:
            enhanced = f"{full_prompt}{quality_tail}"
            width, height = 768, 768

        negative = (
            "blurry, low quality, pixelated, distorted, deformed, ugly, bad anatomy, "
            "watermark, signature, text errors, garbled text, poorly drawn, disfigured, "
            "oversaturated, washed out, extra limbs, duplicate, clone, artifact, noise"
        )

        if NANO_BANANA_API_KEY:
            image_urls = None
            if image_b64:
                try:
                    raw = base64.b64decode(image_b64, validate=True)
                    if len(raw) > MAX_UPLOAD_BYTES:
                        return jsonify({"error": "image too large (max 8MB)"}), 400
                    img_id = _store_temp_image(raw, mime_type)
                    base_url = request.host_url.rstrip('/')
                    image_urls = [f"{base_url}/api/temp-image/{img_id}"]
                except Exception:
                    pass
            try:
                task_id, err = nano_banana_submit(enhanced, image_urls=image_urls)
                if not err:
                    result_url, err = nano_banana_poll(task_id)
                    if not err:
                        img_resp = requests.get(result_url, timeout=30)
                        img_resp.raise_for_status()
                        return jsonify({"image": _finalize_generated_image(img_resp.content, watermark_opts)})
            except Exception:
                pass

        if HF_API_KEY:
            try:
                resp = requests.post(
                    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
                    headers={"Authorization": f"Bearer {HF_API_KEY}"},
                    json={"inputs": enhanced},
                    timeout=90,
                )
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                    return jsonify({"image": _finalize_generated_image(resp.content, watermark_opts)})
            except Exception:
                pass

        seed = int(time.time()) % 99999
        encoded_prompt   = urllib.parse.quote(enhanced)
        encoded_negative = urllib.parse.quote(negative)
        poll_url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width={width}&height={height}&seed={seed}"
            f"&negative={encoded_negative}"
            f"&model=flux&enhance=true&nologo=true"
        )
        img_resp = requests.get(poll_url, timeout=120)
        if img_resp.status_code == 200 and img_resp.headers.get("content-type", "").startswith("image/"):
            return jsonify({"image": _finalize_generated_image(img_resp.content, watermark_opts)})
        return jsonify({"error": f"Image generation failed (status {img_resp.status_code}). Please try again."}), 502

    except Exception as e:
        return jsonify({"error": f"Image generation error: {str(e)}"}), 500


_reengagement_thread_started = False

def _start_reengagement_thread_once():
    """Starts the hourly notification background thread exactly once.
    Render/always-on only — never starts on Vercel (use the cron route)."""
    global _reengagement_thread_started
    if _reengagement_thread_started or IS_SERVERLESS:
        return
    threading.Thread(target=_reengagement_loop, daemon=True).start()
    _reengagement_thread_started = True


_start_reengagement_thread_once()

if __name__ == "__main__":
    active = []
    if PROVIDER in ("auto", "groq") and GROQ_API_KEY:
        active.append(f"Groq({GROQ_MODEL})")
    if PROVIDER in ("auto", "cerebras") and CEREBRAS_API_KEY:
        active.append(f"Cerebras({CEREBRAS_MODEL})")
    providers_str = " → ".join(active) if active else "none configured! (users can still supply their own key in Settings)"
    image_provider = "NanoBanana (image-to-image supported)" if NANO_BANANA_API_KEY else (
        "HuggingFace FLUX (text-to-image only)" if HF_API_KEY else "Pollinations (text-to-image, no key needed)"
    )
    print(f"Starting Mythic AI at http://localhost:5000")
    print(f"Providers (Groq primary, Cerebras fallback): {providers_str}")
    print(f"Image generation: {image_provider}")
    print(f"Re-engagement notifications: Render/always-on -> hourly background "
          f"thread. Vercel -> daily at 12:00 via /api/cron/reengagement (external cron).")
    if IS_SERVERLESS:
        print("NOTE: detected a serverless environment (Vercel) — see the "
              "IS_SERVERLESS comment near the top of this file for the "
              "limitations that come with running this app there.")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
