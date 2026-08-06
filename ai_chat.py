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

Optional — "Paste URL" book/document support:
    pip install beautifulsoup4          (readable text extraction from ordinary webpages)
    pip install playwright pytesseract  (OCR reading of flipbook-style viewers, e.g.
                                          FlippingBook/Issuu/Yumpu/mmdigital-style sites)
    playwright install --with-deps chromium
    Also install the tesseract-ocr system package (e.g. `apt-get install tesseract-ocr`
    in your Render build). Without these, direct PDF/DOCX/TXT links and ordinary
    webpages still work fine — only flipbook OCR is disabled, with a clear message
    telling the user what's missing.

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
import secrets
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
# bs4 — used to pull clean readable text out of ordinary webpages (the
# "Paste URL" box, when the link isn't a direct PDF/DOCX/TXT download).
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
# playwright + pytesseract — power OCR extraction for "flipbook" style book
# viewers (FlippingBook, Issuu, Yumpu, Calameo, AnyFlip, and similar), which
# render pages as images/canvas rather than sending any real text. Both are
# optional: if either is missing, flipbook OCR is silently disabled and the
# app falls back to a clear "can't read this" message instead of crashing.
# Install with:
#   pip install playwright pytesseract
#   playwright install --with-deps chromium
#   apt-get install -y tesseract-ocr   (or the equivalent on your host/Render build)
try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
try:
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
_FLIPBOOK_OCR_AVAILABLE = _PLAYWRIGHT_AVAILABLE and _OCR_AVAILABLE and _WATERMARK_AVAILABLE  # needs PIL too
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

# --- Explicit session cookie config -------------------------------------------
# Without these, Flask falls back to defaults that can behave inconsistently
# across browsers/proxies (Render sits behind a proxy that terminates HTTPS,
# so we need SESSION_COOKIE_SECURE=True but the app itself sees plain HTTP —
# that's fine, the browser still only sends it over https). Setting these
# explicitly avoids the "works in one browser, silently empty in another"
# class of bug caused by a cookie getting dropped or expiring sooner than
# expected.
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=True,       # only sent over https (Render is https)
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=365),
)


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
    # On Vercel (and any serverless host), the deployed script directory is
    # READ-ONLY — only /tmp is writable, and /tmp itself is ephemeral and NOT
    # shared across instances. That combination means this file-based fallback
    # can never actually keep your session stable on Vercel: writes here may
    # raise (read-only fs) or, even if /tmp is used, vanish on the next cold
    # start / different instance, silently minting a new "you" and breaking
    # owner checks like the API keys page. Detect that case explicitly instead
    # of crashing, and fall back to a same-process-lifetime key so at least a
    # single warm instance stays consistent — but this is NOT a real fix for
    # Vercel; set FLASK_SECRET_KEY as an env var there (see comment above).
    is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))
    if is_serverless:
        print("[secret-key] WARNING: FLASK_SECRET_KEY is not set and this is a "
              "serverless (Vercel) deployment. Sessions/owner status WILL NOT "
              "persist reliably across requests. Set FLASK_SECRET_KEY in your "
              "Vercel project's environment variables to fix this.")
        return str(uuid.uuid4())
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "chat_data")
    try:
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
    except OSError as e:
        print(f"[secret-key] Could not persist secret key to disk ({e}); "
              f"using a process-lifetime key instead. Set FLASK_SECRET_KEY "
              f"as an env var for a permanent fix.")
        return str(uuid.uuid4())


app.secret_key = _persistent_secret_key()

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB — images and general attachments
DOCUMENT_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB — books/PDFs/docs specifically

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
    """Each visitor gets a unique anonymous ID, normally stored in their
    browser session cookie. No login required — conversations are private
    per browser/device.

    Resilience: the frontend also keeps a copy of this id in localStorage
    and sends it as the X-Client-Id header on every /api/ request. If the
    session cookie ever gets dropped (cookie banner blocking, a browser
    setting, an intermediate proxy stripping it, etc.) but localStorage
    still has the id, we reseed the session from that header instead of
    silently generating a brand new random user and "losing" the chat
    history. A client-supplied id is only accepted if it looks like a
    valid UUID, so this can't be abused to guess/hijack another user's id.
    """
    if "user_id" not in session:
        client_id = request.headers.get("X-Client-Id", "").strip()
        if client_id:
            try:
                uuid.UUID(client_id)
                session["user_id"] = client_id
            except (ValueError, AttributeError):
                session["user_id"] = str(uuid.uuid4())
        else:
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
        else:
            print(f"[Supabase] list_conversations failed: HTTP {r.status_code} — {r.text[:300]}")
    except Exception as e:
        print(f"[Supabase] list_conversations exception: {e}")
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
        else:
            print(f"[Supabase] load_conversation failed: HTTP {r.status_code} — {r.text[:300]}")
    except Exception as e:
        print(f"[Supabase] load_conversation exception: {e}")
    return None


def save_conversation(username, conv_id, data):
    data["updated_at"] = time.time()
    if not SUPABASE_URL:
        _save_conversation_file(username, conv_id, data)
        return
    try:
        headers = {**sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
        r = requests.post(
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
        if r.status_code not in (200, 201, 204):
            print(f"[Supabase] save_conversation failed: HTTP {r.status_code} — {r.text[:500]}")
    except Exception as e:
        print(f"[Supabase] save_conversation exception: {e}")


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

# --- Permanent, account-wide invite link -------------------------------------
# One stable code, generated once and reused forever after (persisted to
# disk / Supabase so it doesn't change on every restart or redeploy). This
# is what makes the invite link look/behave like an actual generated share
# link (…/invite/<code>) instead of just the bare domain, while still
# always resolving to the same public, no-login chat page for everyone.
_INVITE_CODE_FILE = _os.path.join(_DATA_DIR, "invite_code.txt")
_invite_code_lock = threading.Lock()

def get_or_create_invite_code():
    with _invite_code_lock:
        if SUPABASE_URL:
            try:
                r = requests.get(sb("app_settings?key=eq.invite_code&select=value"),
                                  headers=sb_headers(), timeout=10)
                if r.status_code == 200 and r.json():
                    return r.json()[0]["value"]
            except Exception as e:
                print(f"[Supabase] get_or_create_invite_code read failed: {e} — falling back to local file.")
        if _os.path.exists(_INVITE_CODE_FILE):
            try:
                with open(_INVITE_CODE_FILE, encoding="utf-8") as f:
                    code = f.read().strip()
                    if code:
                        return code
            except Exception:
                pass
        code = uuid.uuid4().hex[:12]
        if SUPABASE_URL:
            try:
                requests.post(sb("app_settings"), headers=sb_headers(),
                              json={"key": "invite_code", "value": code}, timeout=10)
            except Exception as e:
                print(f"[Supabase] get_or_create_invite_code write failed: {e} — falling back to local file.")
        try:
            with open(_INVITE_CODE_FILE, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            print(f"[invite] could not persist invite code to disk: {e}")
        return code

# --- Owner account id ---------------------------------------------------------
# The invite link is meant to share YOUR chat history with whoever opens it
# (not give them their own empty account). So we need one fixed "owner"
# user_id that anyone opening /invite/<code> gets logged into directly.
# Persisted the same way as the invite code so it survives restarts/redeploys.
_OWNER_ID_FILE = _os.path.join(_DATA_DIR, "owner_user_id.txt")
_owner_id_lock = threading.Lock()

# Hard override for serverless hosts (Vercel, etc.) where neither the local
# filesystem (/tmp is wiped on cold start) nor the session cookie can be
# trusted to persist the owner id reliably without Supabase configured.
# Set OWNER_USER_ID to any fixed string (e.g. a UUID you generate once) as
# an environment variable, and that value is used directly, every time, on
# every instance — no file, no database, no race to "claim" ownership.
_OWNER_USER_ID_ENV = _os.environ.get("OWNER_USER_ID", "").strip()

def get_or_create_owner_id(preferred_id=None):
    if _OWNER_USER_ID_ENV:
        return _OWNER_USER_ID_ENV
    with _owner_id_lock:
        if SUPABASE_URL:
            try:
                r = requests.get(sb("app_settings?key=eq.owner_user_id&select=value"),
                                  headers=sb_headers(), timeout=10)
                if r.status_code == 200 and r.json():
                    return r.json()[0]["value"]
            except Exception as e:
                print(f"[owner] Supabase read failed: {e} — falling back to local file.")
        if _os.path.exists(_OWNER_ID_FILE):
            try:
                with open(_OWNER_ID_FILE, encoding="utf-8") as f:
                    oid = f.read().strip()
                    if oid:
                        return oid
            except Exception:
                pass
        # First time this is ever called: prefer adopting the caller's
        # existing session id (so their real chat history becomes the
        # shared "owner" history) rather than minting a brand new empty one.
        oid = preferred_id if preferred_id else str(uuid.uuid4())
        if SUPABASE_URL:
            try:
                requests.post(sb("app_settings"), headers=sb_headers(),
                              json={"key": "owner_user_id", "value": oid}, timeout=10)
            except Exception as e:
                print(f"[owner] Supabase write failed: {e} — falling back to local file.")
        try:
            with open(_OWNER_ID_FILE, "w", encoding="utf-8") as f:
                f.write(oid)
        except Exception as e:
            print(f"[owner] could not persist owner id to disk: {e}")
        return oid

# --- Public API keys ("aarav-...") --------------------------------------------
# Lets other apps/people call YOUR Mythic AI like a hosted API (OpenAI-style),
# authenticated with a personal key instead of the free chat UI. Keys are
# generated as "aarav-<random>", and only a SHA-256 hash is ever stored —
# the plaintext key is shown once at creation time and never again, same as
# how OpenAI/Anthropic/Stripe etc. handle API keys.
_API_KEYS_FILE = _os.path.join(_DATA_DIR, "api_keys.json")
_api_keys_lock = threading.Lock()
API_KEY_PREFIX = "aarav-"

def _hash_api_key(raw_key):
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

def _load_local_api_keys():
    if _os.path.exists(_API_KEYS_FILE):
        try:
            with open(_API_KEYS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_local_api_keys(keys):
    try:
        with open(_API_KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(keys, f)
    except Exception as e:
        print(f"[api_keys] could not persist keys to disk: {e}")

def create_api_key(label="", username=""):
    """Generates a new 'aarav-...' key (total 45 chars), stores only its hash, and returns the
    ONE-TIME plaintext key alongside its public record. Scoped to `username`
    (the creator's session id) so each visitor only ever sees/manages their
    own keys, not everyone else's."""
    # Generate random part: 45 total - 6 (aarav-) = 39 random chars
    random_part = secrets.token_urlsafe(29)[:39]  # 29 bytes base64 ≈ 39 chars
    raw_key = API_KEY_PREFIX + random_part
    record = {
        "id": str(uuid.uuid4()),
        "key_hash": _hash_api_key(raw_key),
        "key_prefix": raw_key,  # full key is short (25 chars), no need to truncate
        "label": (label or "").strip()[:100],
        "username": username,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "active": True,
        "last_used_at": None,
        "request_count": 0,
    }
    with _api_keys_lock:
        if SUPABASE_URL:
            try:
                r = requests.post(sb("api_keys"), headers={**sb_headers(), "Prefer": "return=minimal"},
                                   json=record, timeout=10)
                if r.status_code in (200, 201, 204):
                    return raw_key, record
                print(f"[api_keys] Supabase write failed: HTTP {r.status_code} — {r.text[:300]} "
                      f"— falling back to local file.")
            except Exception as e:
                print(f"[api_keys] Supabase write failed: {e} — falling back to local file.")
        keys = _load_local_api_keys()
        keys.append(record)
        _save_local_api_keys(keys)
        return raw_key, record

def list_api_keys(username=""):
    """Returns key records WITHOUT the hash (never expose that) for display,
    filtered to only the ones the calling user created."""
    if SUPABASE_URL:
        try:
            r = requests.get(
                sb(f"api_keys?select=id,key_prefix,label,created_at,active,last_used_at,request_count"
                   f"&username=eq.{urllib.parse.quote(username)}&order=created_at.desc"),
                headers=sb_headers(), timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"[api_keys] Supabase read failed: {e} — falling back to local file.")
    keys = _load_local_api_keys()
    return [{k: v for k, v in rec.items() if k != "key_hash"}
            for rec in reversed(keys) if rec.get("username") == username]

def revoke_api_key(key_id, username=""):
    """Only revokes the key if it belongs to `username` — prevents one
    visitor from revoking someone else's key by guessing/reusing an id."""
    if SUPABASE_URL:
        try:
            r = requests.patch(
                sb(f"api_keys?id=eq.{key_id}&username=eq.{urllib.parse.quote(username)}"),
                headers=sb_headers(), json={"active": False}, timeout=10)
            return r.status_code in (200, 204)
        except Exception as e:
            print(f"[api_keys] Supabase revoke failed: {e} — falling back to local file.")
    keys = _load_local_api_keys()
    found = False
    for rec in keys:
        if rec["id"] == key_id and rec.get("username") == username:
            rec["active"] = False
            found = True
    if found:
        _save_local_api_keys(keys)
    return found

def rename_api_key(key_id, new_label, username=""):
    """Renames (relabels) a key — only if it belongs to `username`."""
    new_label = (new_label or "").strip()[:100]
    if SUPABASE_URL:
        try:
            r = requests.patch(
                sb(f"api_keys?id=eq.{key_id}&username=eq.{urllib.parse.quote(username)}"),
                headers=sb_headers(), json={"label": new_label}, timeout=10)
            return r.status_code in (200, 204)
        except Exception as e:
            print(f"[api_keys] Supabase rename failed: {e} — falling back to local file.")
    keys = _load_local_api_keys()
    found = False
    for rec in keys:
        if rec["id"] == key_id and rec.get("username") == username:
            rec["label"] = new_label
            found = True
    if found:
        _save_local_api_keys(keys)
    return found

def verify_api_key(raw_key):
    """Returns True and records usage if raw_key is a valid, active key."""
    if not raw_key or not raw_key.startswith(API_KEY_PREFIX):
        return False
    key_hash = _hash_api_key(raw_key)
    now = datetime.datetime.utcnow().isoformat() + "Z"
    if SUPABASE_URL:
        try:
            r = requests.get(sb(f"api_keys?key_hash=eq.{key_hash}&active=eq.true&select=id,request_count"),
                              headers=sb_headers(), timeout=10)
            if r.status_code == 200 and r.json():
                rec = r.json()[0]
                try:
                    requests.patch(sb(f"api_keys?id=eq.{rec['id']}"), headers=sb_headers(),
                                    json={"last_used_at": now, "request_count": (rec.get("request_count") or 0) + 1},
                                    timeout=10)
                except Exception:
                    pass  # usage tracking is best-effort, shouldn't block the request
                return True
            return False
        except Exception as e:
            print(f"[api_keys] Supabase verify failed: {e} — falling back to local file.")
    keys = _load_local_api_keys()
    for rec in keys:
        if rec.get("key_hash") == key_hash and rec.get("active"):
            rec["last_used_at"] = now
            rec["request_count"] = (rec.get("request_count") or 0) + 1
            _save_local_api_keys(keys)
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ─── ANALYTICS & USAGE TRACKING (1000+ lines module) ──────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#
# Comprehensive tracking system for API key users:
# • Per-API-key usage metrics (requests, tokens, models used)
# • Per-user analytics (chat count, message volume, active hours)
# • Per-conversation export (JSON, CSV, HTML)
# • Full-text search with advanced filtering
# • Admin dashboard with user & key management
# • Usage trends and per-model breakdown
# • Rate limiting insights and quota tracking

import json as _json
import csv as _csv
from io import StringIO as _StringIO
from datetime import datetime as _dt, timedelta as _td

_USAGE_DB_FILE = _os.path.join(_DATA_DIR, "usage_metrics.json")
_USAGE_LOCK = threading.Lock()


def _load_usage_metrics():
    """Load all usage metrics from local storage or Supabase."""
    if _os.path.exists(_USAGE_DB_FILE):
        try:
            with open(_USAGE_DB_FILE, encoding="utf-8") as f:
                data = _json.load(f)
                # Convert lists back to proper types
                if isinstance(data.get("daily_stats"), dict):
                    for day, stats in data["daily_stats"].items():
                        if isinstance(stats.get("unique_users"), list):
                            stats["unique_users"] = set(stats["unique_users"])
                return data
        except Exception:
            pass
    return {"api_key_usage": {}, "user_stats": {}, "model_usage": {}, "daily_stats": {}}


def _save_usage_metrics(data):
    """Save usage metrics to persistent storage, converting sets to lists for JSON."""
    try:
        # Convert sets to lists for JSON serialization
        data_copy = _json.loads(_json.dumps(data, default=str))
        if isinstance(data.get("daily_stats"), dict):
            for day, stats in data.get("daily_stats", {}).items():
                if isinstance(stats.get("unique_users"), set):
                    stats["unique_users"] = list(stats["unique_users"])
        with open(_USAGE_DB_FILE, "w", encoding="utf-8") as f:
            _json.dump(data_copy, f, indent=2)
    except Exception as e:
        print(f"[analytics] failed to save usage metrics: {e}")


def track_api_request(key_id, username, model, tokens_in, tokens_out, endpoint, success=True):
    """Record an API request for analytics and rate-limit tracking."""
    with _USAGE_LOCK:
        metrics = _load_usage_metrics()
        now = _dt.utcnow().isoformat()
        today = now.split("T")[0]

        # Per-key usage
        if key_id not in metrics["api_key_usage"]:
            metrics["api_key_usage"][key_id] = {
                "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
                "total_tokens_in": 0, "total_tokens_out": 0, "by_model": {}, "last_updated": now
            }
        key_rec = metrics["api_key_usage"][key_id]
        key_rec["total_requests"] += 1
        if success:
            key_rec["successful_requests"] += 1
            key_rec["total_tokens_in"] += tokens_in
            key_rec["total_tokens_out"] += tokens_out
            if model not in key_rec["by_model"]:
                key_rec["by_model"][model] = {"requests": 0, "tokens_in": 0, "tokens_out": 0}
            key_rec["by_model"][model]["requests"] += 1
            key_rec["by_model"][model]["tokens_in"] += tokens_in
            key_rec["by_model"][model]["tokens_out"] += tokens_out
        else:
            key_rec["failed_requests"] += 1
        key_rec["last_updated"] = now

        # Per-user stats
        if username not in metrics["user_stats"]:
            metrics["user_stats"][username] = {
                "total_requests": 0, "total_messages": 0, "total_conversations": 0,
                "models_used": [], "first_seen": now, "last_active": now
            }
        user_rec = metrics["user_stats"][username]
        user_rec["total_requests"] += 1
        user_rec["total_messages"] += 1
        user_rec["last_active"] = now
        if model not in user_rec["models_used"]:
            user_rec["models_used"].append(model)

        # Daily aggregates
        if today not in metrics["daily_stats"]:
            metrics["daily_stats"][today] = {
                "total_requests": 0, "total_tokens": 0, "unique_users": set(),
                "by_model": {}, "by_endpoint": {}
            }
        daily = metrics["daily_stats"][today]
        daily["total_requests"] += 1
        daily["total_tokens"] += tokens_in + tokens_out
        daily["unique_users"].add(username)

        if model not in daily["by_model"]:
            daily["by_model"][model] = {"requests": 0, "tokens": 0}
        daily["by_model"][model]["requests"] += 1
        daily["by_model"][model]["tokens"] += tokens_in + tokens_out

        if endpoint not in daily["by_endpoint"]:
            daily["by_endpoint"][endpoint] = 0
        daily["by_endpoint"][endpoint] += 1

        _save_usage_metrics(metrics)


def get_usage_report(username=None, key_id=None, start_date=None, end_date=None, days_back=30):
    """Generate a detailed usage report with optional filtering by user, API key, and date range."""
    with _USAGE_LOCK:
        metrics = _load_usage_metrics()

    if not start_date:
        start_date = (_dt.utcnow() - _td(days=days_back)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = _dt.utcnow().strftime("%Y-%m-%d")

    report = {
        "generated_at": _dt.utcnow().isoformat(),
        "filters": {"username": username, "key_id": key_id, "start_date": start_date, "end_date": end_date},
        "summary": {
            "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "total_tokens_in": 0, "total_tokens_out": 0, "by_model": {}, "daily_breakdown": []
        },
        "api_key_metrics": [],
        "user_metrics": []
    }

    # API key metrics
    if key_id and key_id in metrics["api_key_usage"]:
        report["api_key_metrics"].append(metrics["api_key_usage"][key_id])
    elif not key_id:
        report["api_key_metrics"] = list(metrics["api_key_usage"].values())

    # User metrics
    if username and username in metrics["user_stats"]:
        report["user_metrics"].append(metrics["user_stats"][username])
    elif not username:
        report["user_metrics"] = list(metrics["user_stats"].values())

    # Daily breakdown with date filtering
    for day in sorted(metrics["daily_stats"].keys()):
        if day < start_date or day > end_date:
            continue
        stats = metrics["daily_stats"][day]
        report["summary"]["daily_breakdown"].append({"date": day, "stats": stats})
        report["summary"]["total_requests"] += stats.get("total_requests", 0)
        for model, mstats in stats.get("by_model", {}).items():
            if model not in report["summary"]["by_model"]:
                report["summary"]["by_model"][model] = {"requests": 0, "tokens": 0}
            report["summary"]["by_model"][model]["requests"] += mstats.get("requests", 0)
            report["summary"]["by_model"][model]["tokens"] += mstats.get("tokens", 0)

    return report


def search_conversations(username, query="", filters=None):
    """Search conversations by full-text query with advanced filters.
    Filters: {'start_date': '2025-01-01', 'end_date': '2025-01-31', 
              'folder': 'work', 'archived': bool, 'pinned': bool, 'min_messages': int}"""
    filters = filters or {}
    convs = list_conversations(username)
    results = []

    for conv in convs:
        # Apply boolean filters
        if filters.get("archived") is not None and conv.get("archived") != filters["archived"]:
            continue
        if filters.get("pinned") is not None and conv.get("pinned") != filters["pinned"]:
            continue
        if filters.get("folder") and conv.get("folder") != filters["folder"]:
            continue

        # Message count filter
        msg_count = len(conv.get("messages", []))
        if filters.get("min_messages") and msg_count < filters["min_messages"]:
            continue

        # Date range filter
        if filters.get("start_date"):
            conv_date = conv.get("updated_at", "")
            if conv_date and conv_date < filters["start_date"]:
                continue
        if filters.get("end_date"):
            conv_date = conv.get("updated_at", "")
            if conv_date and conv_date > filters["end_date"]:
                continue

        # Full-text search across title and messages
        if query:
            query_lower = query.lower()
            search_text = (conv.get("title") or "").lower()
            found = query_lower in search_text

            if not found:
                for msg in conv.get("messages", []):
                    msg_text = (msg.get("text") or "").lower()
                    if query_lower in msg_text:
                        found = True
                        break
            
            if not found:
                continue

        results.append(conv)

    return results


def export_conversation(conv_id, username, format_type="json"):
    """Export a single conversation in requested format (json, csv, html).
    Returns (content_bytes, mimetype, filename) tuple."""
    conv = load_conversation(username, conv_id)
    if not conv:
        return None, None, None

    title = conv.get("title", "chat")
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)[:50]

    if format_type == "json":
        content = _json.dumps(conv, indent=2, default=str).encode("utf-8")
        return content, "application/json", f"{safe_title}.json"

    elif format_type == "csv":
        output = _StringIO()
        writer = _csv.writer(output)
        writer.writerow(["Role", "Timestamp", "Message"])
        for msg in conv.get("messages", []):
            role = "User" if msg.get("role") == "user" else "AI"
            parts = msg.get("parts", [])
            text = "".join(part.get("text", "") for part in parts if "text" in part)
            writer.writerow([role, msg.get("created_at", ""), text])
        content = output.getvalue().encode("utf-8")
        return content, "text/csv", f"{safe_title}.csv"

    elif format_type == "html":
        html_parts = [
            f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_esc_html(title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; background: #fafafa; }}
    h1 {{ color: #10a37f; border-bottom: 3px solid #10a37f; padding-bottom: 10px; margin-top: 0; }}
    .meta {{ font-size: 0.9em; color: #666; margin: 10px 0 30px; }}
    .msg {{ margin: 15px 0; padding: 12px 16px; border-radius: 8px; }}
    .msg.user {{ background: #e8f5e9; margin-left: 40px; border-left: 3px solid #10a37f; }}
    .msg.ai {{ background: #f5f5f5; margin-right: 40px; border-left: 3px solid #ccc; }}
    .role {{ font-weight: 700; font-size: 0.9em; color: #555; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .content {{ margin-bottom: 6px; white-space: pre-wrap; word-wrap: break-word; }}
    .timestamp {{ font-size: 0.8em; color: #999; }}
    footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #999; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>{_esc_html(title)}</h1>
  <div class="meta">
    <p>Exported on {_dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | Total messages: {len(conv.get("messages", []))}</p>
  </div>
"""
        ]

        for msg in conv.get("messages", []):
            role = "User" if msg.get("role") == "user" else "AI"
            role_class = "user" if msg.get("role") == "user" else "ai"
            parts = msg.get("parts", [])
            text = "".join(part.get("text", "") for part in parts if "text" in part)
            safe_text = _esc_html(text)
            ts = msg.get("created_at", "")
            html_parts.append(f'<div class="msg {role_class}"><div class="role">{role}</div><div class="content">{safe_text}</div><div class="timestamp">{ts}</div></div>')

        html_parts.append(f"""
  <footer>
    <p>This conversation was exported from Mythic AI. It contains {len(conv.get("messages", []))} messages.</p>
  </footer>
</body>
</html>""")
        
        content = "".join(html_parts).encode("utf-8")
        return content, "text/html", f"{safe_title}.html"

    return None, None, None


def _esc_html(s):
    """Escape HTML special characters."""
    if not s:
        return ""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                   .replace('"', "&quot;").replace("'", "&#39;"))


def get_admin_stats(include_daily=False):
    """Return aggregate admin-level statistics (super-admin only).
    Returns overall system metrics, per-key usage, per-user stats."""
    with _USAGE_LOCK:
        metrics = _load_usage_metrics()
    
    total_requests = sum(k.get("total_requests", 0) for k in metrics.get("api_key_usage", {}).values())
    total_tokens_in = sum(k.get("total_tokens_in", 0) for k in metrics.get("api_key_usage", {}).values())
    total_tokens_out = sum(k.get("total_tokens_out", 0) for k in metrics.get("api_key_usage", {}).values())
    
    # Top models by usage
    top_models = {}
    for key_data in metrics.get("api_key_usage", {}).values():
        for model, stats in key_data.get("by_model", {}).items():
            if model not in top_models:
                top_models[model] = {"requests": 0, "tokens": 0}
            top_models[model]["requests"] += stats.get("requests", 0)
            top_models[model]["tokens"] += stats.get("tokens_in", 0) + stats.get("tokens_out", 0)
    
    stats = {
        "total_api_keys_issued": len(metrics.get("api_key_usage", {})),
        "total_users": len(metrics.get("user_stats", {})),
        "all_time_requests": total_requests,
        "all_time_tokens_in": total_tokens_in,
        "all_time_tokens_out": total_tokens_out,
        "all_time_tokens_total": total_tokens_in + total_tokens_out,
        "top_models_by_requests": sorted(top_models.items(), key=lambda x: x[1]["requests"], reverse=True)[:10],
        "api_key_metrics": metrics.get("api_key_usage", {}),
        "user_metrics": metrics.get("user_stats", {}),
        "generated_at": _dt.utcnow().isoformat()
    }
    
    if include_daily:
        stats["daily_stats"] = metrics.get("daily_stats", {})
    
    return stats


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


# ── Public share links ────────────────────────────────────────────────────
# A "share" maps a short public id -> (username, conv_id) so anyone with the
# link can view a read-only copy of that one conversation at /share/<id>,
# with no login and no access to the rest of that person's chats. Uses
# Supabase (a `shares` table) when configured, else a local JSON index file
# alongside the conversation JSON fallback.
_SHARES_INDEX_FILE = _os.path.join(_DATA_DIR, "shares_index.json")
_shares_lock = threading.Lock()


def _load_shares_index():
    try:
        if _os.path.exists(_SHARES_INDEX_FILE):
            with open(_SHARES_INDEX_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_shares_index(data):
    try:
        with open(_SHARES_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[shares] failed to save shares index: {e}")


def get_active_share_id(username, conv_id):
    """Returns the existing active (non-revoked) share id for this
    conversation, or None if it has never been shared / was revoked.
    Checks Supabase first (if configured), then always falls back to
    checking the local file index too — a share may have landed there if
    Supabase writes were failing (e.g. the `shares` table doesn't exist
    yet) when it was created. See create_share_link()."""
    if SUPABASE_URL:
        try:
            r = requests.get(
                sb(f"shares?username=eq.{username}&conv_id=eq.{conv_id}&revoked=eq.false&select=id"),
                headers=sb_headers(), timeout=10,
            )
            if r.status_code == 200 and r.json():
                return r.json()[0]["id"]
            elif r.status_code != 200:
                print(f"[Supabase] get_active_share_id: HTTP {r.status_code} — {r.text[:200]}")
        except Exception as e:
            print(f"[Supabase] get_active_share_id exception: {e}")

    with _shares_lock:
        idx = _load_shares_index()
    for sid, rec in idx.items():
        if rec.get("username") == username and rec.get("conv_id") == conv_id and not rec.get("revoked"):
            return sid
    return None


def create_share_link(username, conv_id):
    """Creates (or reuses, if one is already active) a public share id for
    conv_id owned by username. Returns the share_id string.

    IMPORTANT: if Supabase is configured but the write fails for any reason
    (most commonly: the `shares` table hasn't been created yet), this does
    NOT hand back a link that silently 404s — it transparently falls back
    to the local JSON index instead, so the link that's returned always
    actually works."""
    existing = get_active_share_id(username, conv_id)
    if existing:
        return existing

    share_id = uuid.uuid4().hex[:12]
    if SUPABASE_URL:
        try:
            r = requests.post(
                sb("shares"),
                headers={**sb_headers(), "Prefer": "return=minimal"},
                json={"id": share_id, "username": username, "conv_id": conv_id,
                      "revoked": False, "created_at": time.time()},
                timeout=10,
            )
            if r.status_code in (200, 201, 204):
                return share_id
            print(f"[Supabase] create_share_link failed: HTTP {r.status_code} — {r.text[:300]}"
                  f" — falling back to local storage for this share link.")
        except Exception as e:
            print(f"[Supabase] create_share_link exception: {e} — falling back to local storage.")

    with _shares_lock:
        idx = _load_shares_index()
        idx[share_id] = {"username": username, "conv_id": conv_id,
                          "revoked": False, "created_at": time.time()}
        _save_shares_index(idx)
    return share_id


def resolve_share_link(share_id):
    """Returns {"username":..., "conv_id":...} for an active (non-revoked)
    share id, or None if it doesn't exist / has been revoked. Checks
    Supabase first (if configured), then always checks the local file
    index too, since create_share_link() may have fallen back to it."""
    if SUPABASE_URL:
        try:
            r = requests.get(
                sb(f"shares?id=eq.{share_id}&select=username,conv_id,revoked"),
                headers=sb_headers(), timeout=10,
            )
            if r.status_code == 200:
                rows = r.json()
                if rows:
                    if rows[0].get("revoked"):
                        return None
                    return {"username": rows[0]["username"], "conv_id": rows[0]["conv_id"]}
            else:
                print(f"[Supabase] resolve_share_link: HTTP {r.status_code} — {r.text[:200]}")
        except Exception as e:
            print(f"[Supabase] resolve_share_link exception: {e}")

    idx = _load_shares_index()
    rec = idx.get(share_id)
    if rec and not rec.get("revoked"):
        return {"username": rec["username"], "conv_id": rec["conv_id"]}
    return None


def revoke_share_link(username, conv_id):
    """Revokes any active share link(s) for this conversation, in both
    Supabase and the local index (a share may live in either, depending on
    whether Supabase writes were working when it was created)."""
    if SUPABASE_URL:
        try:
            requests.patch(
                sb(f"shares?username=eq.{username}&conv_id=eq.{conv_id}"),
                headers=sb_headers(), json={"revoked": True}, timeout=10,
            )
        except Exception as e:
            print(f"[Supabase] revoke_share_link exception: {e}")
    with _shares_lock:
        idx = _load_shares_index()
        changed = False
        for sid, rec in idx.items():
            if rec.get("username") == username and rec.get("conv_id") == conv_id and not rec.get("revoked"):
                rec["revoked"] = True
                changed = True
        if changed:
            _save_shares_index(idx)


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
  #api-keys-shortcut-btn { margin:0 12px 6px; padding:9px 14px; background:none; border:1px solid var(--border);
    color:var(--muted); border-radius:8px; font-size:13px; font-weight:500; cursor:pointer;
    text-align:left; touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #api-keys-shortcut-btn:hover { background:var(--panel); color:var(--text); border-color:var(--accent); }
  #conv-list { flex:1; overflow-y:auto; padding:0 8px; display:flex; flex-direction:column; gap:2px; }
  .conv-item { display:flex; align-items:center; justify-content:space-between; gap:6px;
    padding:9px 10px; border-radius:7px; cursor:pointer; font-size:13px; color:var(--muted); }
  .conv-item:hover { background:var(--accent-dim); color:var(--text); }
  .conv-item.active { background:var(--accent-dim); color:var(--accent); font-weight:500; }
  .conv-item .title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
  .conv-item .menu-btn { opacity:0; background:none; border:none; color:var(--muted);
    cursor:pointer; font-size:16px; padding:2px 8px; border-radius:5px; flex-shrink:0;
    touch-action:manipulation; }
  .conv-item:hover .menu-btn { opacity:1; }
  .conv-item .menu-btn:hover { color:var(--accent); background:rgba(255,255,255,.06); }
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
  #share-btn { background:none; border:1px solid var(--border); color:var(--muted);
    width:36px; height:36px; border-radius:6px; cursor:pointer; font-size:15px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; touch-action:manipulation; }
  #share-btn:hover { background:var(--panel); }
  #share-btn.active { color:var(--accent); border-color:var(--accent); }

  #share-modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.6);
    z-index:250; align-items:center; justify-content:center; padding:16px; }
  #share-modal-overlay.show { display:flex; }
  #share-modal { background:var(--bg); border:1px solid var(--border); border-radius:14px;
    padding:22px; width:100%; max-width:400px; box-shadow:0 10px 40px rgba(0,0,0,.3); }
  #share-modal h3 { margin:0 0 4px; font-size:16px; }
  #share-modal p.sub { margin:0 0 16px; font-size:12.5px; color:var(--muted); }
  #share-link-row { display:flex; gap:8px; margin-bottom:14px; }
  #share-link-input { flex:1; min-width:0; padding:9px 12px; border-radius:8px;
    border:1.5px solid var(--border); background:var(--panel); color:var(--text);
    font-size:12.5px; outline:none; text-overflow:ellipsis; }
  #share-link-input:focus { border-color:var(--accent); }
  #share-open-btn { background:var(--panel); border:1px solid var(--border); color:var(--muted);
    border-radius:8px; padding:0 12px; font-size:14px; cursor:pointer; flex-shrink:0; }
  #share-open-btn:hover { border-color:var(--accent); color:var(--accent); }
  #share-copy-btn { background:var(--accent); color:#fff; border:none; border-radius:8px;
    padding:0 14px; font-size:12.5px; font-weight:700; cursor:pointer; flex-shrink:0; }
  #share-copy-btn:hover { opacity:.9; }
  #share-native-btn { width:100%; background:var(--panel); border:1px solid var(--border);
    color:var(--text); border-radius:8px; padding:10px; font-size:13px; cursor:pointer;
    font-family:inherit; margin-bottom:8px; }
  #share-native-btn:hover { border-color:var(--accent); color:var(--accent); }
  #share-revoke-btn { width:100%; background:none; border:1px solid var(--border);
    color:#ef4444; border-radius:8px; padding:10px; font-size:12.5px; cursor:pointer;
    font-family:inherit; margin-bottom:8px; }
  #share-revoke-btn:hover { background:rgba(239,68,68,.08); }
  #share-close-btn { width:100%; background:none; border:1px solid var(--border);
    color:var(--muted); border-radius:8px; padding:10px; font-size:13px; cursor:pointer;
    font-family:inherit; }
  #share-status { font-size:11.5px; color:var(--muted); text-align:center; margin-top:2px; min-height:14px; }
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
    #share-btn { width:36px; height:36px; font-size:13px; }
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
    .conv-item .menu-btn { opacity:1; }
    #sidebar-footer { font-size:11px; padding:10px 12px; }
  }

  @media(max-width:380px) {
    :root { --sidebar-w: 88vw; }
    .msg { font-size:13.5px; }
    header h1 { font-size:13px; }
    #speak-toggle { display:none; }
  }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
</head>
<body>
<div class="layout">
  <div id="sidebar-overlay" style="display:none;position:fixed;inset:0;background:#0007;z-index:99"></div>
  <div id="sidebar">
    <button id="new-chat-btn">+ New chat</button>
    <button id="api-keys-shortcut-btn" title="Manage API keys">🔑 API Keys</button>
    <div id="api-usage-summary" style="display:none;margin:0 12px 6px;padding:8px 10px;border:1px solid var(--border);border-radius:8px;font-size:11.5px;color:var(--muted);cursor:pointer;" title="Click to manage API keys"></div>
    <div style="display:flex;gap:6px;margin:6px 0;">
      <button id="search-chats-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:7px 4px;font-size:12px;cursor:pointer;font-family:inherit;">🔎 Search</button>
      <button id="reminders-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:7px 4px;font-size:12px;cursor:pointer;font-family:inherit;">⏰ Reminders</button>
    </div>
    <div id="conv-list"></div>
    <div id="sidebar-footer">
      <div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;">
        <button id="archived-toggle-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:6px;font-size:11px;cursor:pointer;font-family:inherit;">⭐ Starred</button>
        <button id="bookmarks-btn" style="flex:1;background:none;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:6px;font-size:11px;cursor:pointer;font-family:inherit;">🔖 Bookmarks</button>
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
        <button id="share-btn" title="Get invite link">🔗</button>
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
      <button class="quick-btn" id="homework-btn">📚 Homework & Study</button>
      <button class="quick-btn" id="weather-btn">🌤 Weather</button>
      <button class="quick-btn" id="search-btn">🔍 Search</button>
      <button class="quick-btn" id="code-workspace-btn">💻 Code</button>
    </div>
      <form id="chat-form">
        <div class="input-row">
          <input type="file" id="file-input" accept="image/*,.txt,.md,.csv,.json,.pdf,.docx" style="display:none">
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

<div id="share-modal-overlay">
  <div id="share-modal">
    <h3>🔗 Invite link</h3>
    <p class="sub">One permanent link for the whole app — not tied to any single chat. Share it with anyone; each person who opens it gets their own private conversation with Mythic AI, no login required.</p>
    <div id="share-link-row">
      <input type="text" id="share-link-input" readonly>
      <button id="share-open-btn" type="button" title="Open in a new tab">↗</button>
      <button id="share-copy-btn" type="button">Copy</button>
    </div>
    <div id="share-qr-wrap" style="display:flex;justify-content:center;margin:16px 0;">
      <div id="share-qr-box" style="position:relative;width:220px;height:220px;background:#fff;padding:12px;border-radius:12px;">
        <div id="share-qr-canvas" style="width:100%;height:100%;"></div>
        <div id="share-qr-logo" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
             width:52px;height:52px;background:#fff;border-radius:12px;display:flex;align-items:center;
             justify-content:center;box-shadow:0 0 0 4px #fff;">
          <img src="/icon.png" alt="Mythic AI" style="width:40px;height:40px;border-radius:8px;object-fit:cover;">
        </div>
      </div>
    </div>
    <button id="share-native-btn" type="button">📤 Share via…</button>
    <button id="share-revoke-btn" type="button">Stop sharing</button>
    <button id="share-close-btn" type="button">Close</button>
    <div id="share-status"></div>
  </div>
</div>

<div id="api-usage-overlay" style="display:none;position:fixed;inset:0;background:#0f1115;color:#f2f2f2;z-index:200;overflow-y:auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:1000px;margin:0 auto;padding:32px 24px 60px;">
    <button id="api-usage-close-btn" style="background:none;border:none;color:#9a9ea6;font-size:14px;cursor:pointer;padding:0;margin-bottom:20px;font-family:inherit;">← Back to chat</button>
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px;margin-bottom:8px;">
      <div>
        <h1 style="font-size:30px;margin:0 0 6px;">API keys</h1>
        <div style="color:#9a9ea6;font-size:14px;margin-bottom:30px;max-width:560px;line-height:1.5;">Create and manage API keys for authenticating requests to Mythic AI. These keys allow programmatic access to your app.</div>
      </div>
      <button id="api-usage-gen-btn" style="background:#e8532a;color:#fff;border:none;border-radius:8px;padding:11px 18px;font-size:13px;font-weight:700;letter-spacing:.3px;cursor:pointer;white-space:nowrap;">GENERATE API KEY</button>
    </div>
    <div id="api-usage-totals" style="display:flex;gap:16px;margin-bottom:30px;flex-wrap:wrap;"></div>
    <table style="width:100%;border-collapse:collapse;background:#1a1d24;border:1px solid #2a2e37;border-radius:14px;overflow:hidden;">
      <thead><tr>
        <th style="text-align:left;font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#9a9ea6;padding:14px 16px;border-bottom:1px solid #2a2e37;">Name</th>
        <th style="text-align:left;font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#9a9ea6;padding:14px 16px;border-bottom:1px solid #2a2e37;">API Key</th>
        <th style="text-align:left;font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#9a9ea6;padding:14px 16px;border-bottom:1px solid #2a2e37;">Created At</th>
        <th style="text-align:left;font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#9a9ea6;padding:14px 16px;border-bottom:1px solid #2a2e37;">Calls</th>
        <th style="text-align:left;font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#9a9ea6;padding:14px 16px;border-bottom:1px solid #2a2e37;">State</th>
        <th style="text-align:left;font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#9a9ea6;padding:14px 16px;border-bottom:1px solid #2a2e37;">Options</th>
      </tr></thead>
      <tbody id="api-usage-tbody"></tbody>
    </table>
    <div id="api-usage-empty" style="display:none;color:#9a9ea6;font-size:15px;padding:50px 0;text-align:center;">No API keys yet. Click "Generate API Key" to create one.</div>
  </div>

  <div id="api-usage-create-overlay" style="display:none;position:fixed;inset:0;background:#000a;align-items:center;justify-content:center;z-index:210;">
    <div style="background:#1a1d24;border:1px solid #2a2e37;border-radius:14px;padding:26px;width:min(90vw,420px);">
      <h3 style="margin:0 0 14px;font-size:18px;">Generate API key</h3>
      <input type="text" id="api-usage-create-label" placeholder="Key name (optional)" maxlength="100" style="width:100%;padding:10px 12px;border-radius:8px;border:1px solid #3a3e47;background:#0f1115;color:#fff;font-size:14px;margin-bottom:14px;">
      <div id="api-usage-new-key-result"></div>
      <div style="display:flex;gap:10px;justify-content:flex-end;">
        <button id="api-usage-create-close-btn" style="border-radius:8px;padding:9px 16px;font-size:13px;cursor:pointer;border:none;background:#2a2e37;color:#f2f2f2;">Close</button>
        <button id="api-usage-create-confirm-btn" style="border-radius:8px;padding:9px 16px;font-size:13px;cursor:pointer;border:none;background:#e8532a;color:#fff;font-weight:700;">Generate</button>
      </div>
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

    <div class="settings-section" id="api-keys-section" style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px;">
      <label>🔑 API Keys</label>
      <div class="hint">Let other apps call Mythic AI like an OpenAI-style API. Keys start with
        <code>aarav-</code> and are shown only once — copy them right away.</div>
      <div style="display:flex;gap:8px;margin-top:8px;">
        <input type="text" id="api-key-label-input" class="settings-text-input"
          placeholder="Label (e.g. 'BattleZoneApp')" style="flex:1;">
        <button type="button" id="api-key-create-btn" class="settings-btn">+ Generate</button>
      </div>
      <div id="api-key-new-box" style="display:none;margin-top:10px;padding:10px;border:1px solid var(--accent);border-radius:8px;background:var(--bg);">
        <div style="font-size:12px;opacity:.8;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
          <span>New key — copy it now, it won't be shown again:</span>
          <button type="button" id="api-key-copy-btn" style="background:var(--accent);color:#000;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px;font-weight:700;white-space:nowrap;">Copy</button>
        </div>
        <input type="text" id="api-key-new-value" readonly style="width:100%;padding:10px;font-family:monospace;font-size:11px;letter-spacing:0.5px;background:var(--panel);border:1px solid var(--border);border-radius:6px;color:var(--text);box-sizing:border-box;cursor:text;overflow:auto;white-space:nowrap;" />
      </div>
      <div id="api-key-list" style="margin-top:10px;display:flex;flex-direction:column;gap:6px;"></div>
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

    <div class="settings-section" style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px;">
      <label>💾 Backup all chats</label>
      <p style="font-size:11.5px;color:var(--muted);margin:2px 0 8px;">Download every conversation as one file, or restore from a previous backup — also the only way to carry your chats over to a different browser/device, since accounts here are anonymous per-browser.</p>
      <div style="display:flex;gap:8px;">
        <button id="backup-export-btn" type="button" style="flex:1;background:none;border:1px solid var(--border);color:var(--text);border-radius:8px;padding:9px;cursor:pointer;font-family:inherit;font-size:12.5px;">⬇ Export all</button>
        <button id="backup-import-btn" type="button" style="flex:1;background:none;border:1px solid var(--border);color:var(--text);border-radius:8px;padding:9px;cursor:pointer;font-family:inherit;font-size:12.5px;">⬆ Import backup</button>
      </div>
      <input id="backup-import-file" type="file" accept=".zip" style="display:none;">
      <div id="backup-status" style="font-size:11.5px;color:var(--muted);margin-top:6px;"></div>
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

<!-- ─── HOMEWORK & STUDY BOOK — question, upload, or URL ───────────────────── -->
<div id="homework-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:300;align-items:center;justify-content:center;padding:16px;">
  <div style="background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px;width:92%;max-width:460px;max-height:90vh;overflow-y:auto;">
    <h3 style="margin:0 0 4px;font-size:18px;">📚 Homework &amp; Study</h3>
    <p style="color:var(--muted);font-size:13px;margin:0 0 16px;">Ask a question, upload a book/PDF, or paste a link to one — or all three.</p>

    <textarea id="homework-question" rows="3" placeholder="e.g. Help me with question 4 on quadratic equations, or leave blank if you're just uploading a book"
      style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:13px;outline:none;margin-bottom:12px;font-family:inherit;resize:vertical;"></textarea>

    <div style="display:flex;gap:6px;margin-bottom:10px;">
      <button id="hw-mode-upload" class="hw-mode-btn" data-mode="upload" style="flex:1;padding:9px;border-radius:8px;border:1.5px solid var(--accent);background:var(--accent-dim);color:var(--accent);cursor:pointer;font-size:12.5px;font-family:inherit;">📎 Upload File</button>
      <button id="hw-mode-url" class="hw-mode-btn" data-mode="url" style="flex:1;padding:9px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;font-size:12.5px;font-family:inherit;">🔗 Paste URL</button>
    </div>

    <div id="hw-upload-area">
      <div id="hw-upload-dropzone" style="border:2px dashed var(--border);border-radius:12px;padding:18px;text-align:center;cursor:pointer;margin-bottom:8px;">
        <div style="font-size:13px;color:var(--muted);">📄 Click to choose a PDF, DOCX, or text file</div>
      </div>
      <input type="file" id="hw-file-input" accept=".pdf,.docx,.txt,.md,.csv,.json" style="display:none">
      <div id="hw-file-name" style="font-size:12px;color:var(--accent);display:none;margin-bottom:8px;"></div>
    </div>

    <div id="hw-url-area" style="display:none;">
      <input id="hw-url-input" type="text" placeholder="https://example.com/book.pdf"
        style="width:100%;background:var(--bg);border:1.5px solid var(--border);color:var(--text);border-radius:8px;padding:10px 12px;font-size:13px;outline:none;margin-bottom:8px;font-family:inherit;">
    </div>

    <div id="hw-loading" style="display:none;text-align:center;padding:14px;color:var(--muted);font-size:13px;">Fetching document...</div>
    <div id="hw-error" style="display:none;color:#ef4444;font-size:12px;margin-bottom:8px;padding:8px;background:#fef2f2;border-radius:6px;"></div>

    <div style="display:flex;gap:8px;margin-top:6px;">
      <button id="hw-send-btn" style="flex:1;background:linear-gradient(135deg,#10a37f,#0d7a5f);color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;">Send</button>
      <button id="hw-close-btn" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:10px;padding:12px 16px;font-size:14px;cursor:pointer;">✕</button>
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
// ─── Resilient identity: keep a copy of our anonymous id outside the cookie ──
// If the session cookie ever fails to persist in a given browser (blocked,
// stripped by a proxy, cleared, etc.), this localStorage id lets the server
// reseed the same account instead of silently starting a fresh empty one.
(function () {
  try {
    let cid = localStorage.getItem('mythic_client_id');
    if (!cid) {
      cid = (crypto.randomUUID ? crypto.randomUUID() :
        'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
          const r = Math.random() * 16 | 0;
          return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        }));
      localStorage.setItem('mythic_client_id', cid);
    }
    const _origFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      if (url.startsWith('/api/')) {
        init = init || {};
        init.headers = new Headers(init.headers || {});
        init.headers.set('X-Client-Id', cid);
      }
      return _origFetch(input, init);
    };
  } catch (e) {
    // localStorage unavailable (rare, e.g. some locked-down browser modes) —
    // app still works, just without the cookie-loss fallback.
    console.warn('Client-id resilience layer unavailable:', e);
  }
})();

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

let showingStarredOnly = false;

function buildConvItem(c) {
  const item = document.createElement('div');
  item.className = 'conv-item' + (c.id === activeConvId ? ' active' : '');
  item.innerHTML = '<span class="title"></span>'
    + '<button class="menu-btn" title="More">⋮</button>';
  item.querySelector('.title').textContent = (c.pinned ? '⭐ ' : '') + c.title;

  item.addEventListener('click', (e) => {
    if (e.target.closest('.menu-btn')) return;
    openConversation(c.id);
  });

  item.querySelector('.menu-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    document.querySelectorAll('.conv-menu-dropdown').forEach(el => el.remove());
    const menu = document.createElement('div');
    menu.className = 'conv-menu-dropdown';
    menu.style.cssText = 'position:fixed;background:var(--panel);border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,.3);z-index:400;min-width:170px;overflow:hidden;';
    const rect = e.target.getBoundingClientRect();
    menu.style.top = (rect.bottom + 4) + 'px';
    menu.style.left = Math.max(8, rect.right - 180) + 'px';

    const menuItems = [
      { label: (c.pinned ? '⭐ Unstar' : '☆ Star'), action: async () => {
          await fetch('/api/conversations/' + c.id, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pinned: !c.pinned }) });
          loadConversationList();
        } },
      { label: '⎘ Duplicate', action: async () => {
          const r = await fetch('/api/conversations/' + c.id + '/duplicate', { method: 'POST' });
          const d = await r.json();
          if (d.id) openConversation(d.id);
        } },
      { label: '📁 Move to folder', action: async () => {
          const folders = await fetch('/api/folders').then(r => r.json()).then(d => d.folders || []).catch(() => []);
          const hint = folders.length ? ('Existing: ' + folders.join(', ') + '\n\n') : '';
          const name = prompt(hint + 'Folder name (blank to remove from folder):', c.folder || '');
          if (name === null) return;
          await fetch('/api/conversations/' + c.id, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ folder: name.trim() }) });
          loadConversationList();
        } },
      { label: '✎ Rename', action: async () => {
          const newTitle = prompt('Rename chat:', c.title);
          if (!newTitle || !newTitle.trim() || newTitle.trim() === c.title) return;
          await fetch('/api/conversations/' + c.id, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: newTitle.trim() }) });
          loadConversationList();
        } },
      { label: '🔗 Share link', action: () => openShareModalFor(c.id) },
      { label: (c.archived ? '📤 Unarchive' : '🗄 Archive'), action: async () => {
          await fetch('/api/conversations/' + c.id, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ archived: !c.archived }) });
          if (c.id === activeConvId) startNewChat(); else loadConversationList();
        } },
      { label: '✕ Delete', danger: true, action: async () => {
          await fetch('/api/conversations/' + c.id, { method: 'DELETE' });
          if (c.id === activeConvId) startNewChat(); else loadConversationList();
        } },
    ];
    menuItems.forEach(it => {
      const row = document.createElement('div');
      row.textContent = it.label;
      row.style.cssText = 'padding:9px 14px;font-size:12.5px;cursor:pointer;color:' + (it.danger ? '#ef4444' : 'var(--text)') + ';';
      row.addEventListener('mouseenter', () => row.style.background = 'var(--accent-dim)');
      row.addEventListener('mouseleave', () => row.style.background = '');
      row.addEventListener('click', () => { it.action(); menu.remove(); });
      menu.appendChild(row);
    });
    document.body.appendChild(menu);
    setTimeout(() => {
      document.addEventListener('click', function closeConvMenu() {
        menu.remove();
        document.removeEventListener('click', closeConvMenu);
      }, { once: true });
    }, 0);
  });

  return item;
}

async function loadConversationList() {
  try {
    const r = await fetch('/api/conversations?archived=0');
    const d = await r.json();
    let convs = d.conversations || [];
    if (showingStarredOnly) convs = convs.filter(c => c.pinned);
    convListEl.innerHTML = '';

    if (!showingStarredOnly) {
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
      empty.textContent = showingStarredOnly ? 'No starred chats yet.' : 'No chats yet.';
      empty.style.cssText = 'padding:16px 10px;font-size:12.5px;color:var(--muted);text-align:center;';
      convListEl.appendChild(empty);
    }
    return convs;
  } catch { return []; }
}

async function openConversation(convId, opts) {
  activeConvId = convId;
  // Reflect the open chat in the address bar (?c=<id>) so the URL at the
  // top actually changes per-chat, refreshing the page reopens the same
  // chat, and the browser Back/Forward buttons move between chats.
  // opts.updateUrl=false is used when we're reacting to a popstate event
  // (the URL already matches — pushing again would break Back/Forward).
  if (!opts || opts.updateUrl !== false) {
    try { history.pushState({ conv: convId }, '', '?c=' + encodeURIComponent(convId)); } catch {}
  }
  try {
    const r = await fetch('/api/conversations/' + convId);
    if (!r.ok) return;
    const d = await r.json();
    messagesEl.innerHTML = '';
    (d.messages || []).forEach(m => addMessage(m.role, m.text, m.attachment));
    loadConversationList();
  } catch {}
  refreshShareBtnState();
  if (isMobile()) closeSidebar();
}

function startNewChat(opts) {
  activeConvId = null;
  messagesEl.innerHTML = '';
  showEmptyState();
  refreshShareBtnState();
  if (!opts || opts.updateUrl !== false) {
    try { history.pushState({}, '', location.pathname); } catch {}
  }
  // Create the conversation on the server right away so it shows up in the
  // sidebar immediately, instead of only appearing after the first message.
  fetch('/api/conversations', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      if (d.id) activeConvId = d.id;
      loadConversationList();
    })
    .catch(() => { loadConversationList(); }); // still refresh list even if this failed
}

// Keep the address bar in sync with Back/Forward navigation between chats.
window.addEventListener('popstate', () => {
  const id = new URLSearchParams(location.search).get('c');
  if (id) openConversation(id, { updateUrl: false });
  else startNewChat({ updateUrl: false });
});

function refreshShareBtnState() {
  // The invite link is static and account-wide now (not per-conversation),
  // so there's no per-chat "shared" state to check on the server anymore.
  // Kept as a function (rather than removing every call site) so nothing
  // else in the file needs to change.
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
      let errMsg = 'Something went wrong. Try again.';
      try {
        const errData = await r.clone().json();
        if (errData && errData.error) errMsg = errData.error;
      } catch {
        try {
          const errText = await r.clone().text();
          if (errText && errText.trim()) errMsg = errText.trim().slice(0, 300);
        } catch {}
      }
      addMessage('error', errMsg + ` (HTTP ${r.status})`);
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
  input.value = '';
  input.style.height = 'auto';
  // Cowork mode runs a real multi-step task (plan → work each step →
  // synthesize) via /api/cowork/run instead of a single streamed reply —
  // see runCoworkTask(). Everything else keeps the normal streaming flow.
  if (typeof currentMode !== 'undefined' && currentMode === 'cowork' && text) {
    runCoworkTask(text);
    return;
  }
  addMessage('user', text, attachment);
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

// ─── Share link (per-chat, e.g. mythic-ai.app/share/abc123) ────────────────
const shareBtn          = document.getElementById('share-btn');
const shareModalOverlay = document.getElementById('share-modal-overlay');
const shareLinkInput    = document.getElementById('share-link-input');
const shareOpenBtn      = document.getElementById('share-open-btn');
const shareCopyBtn      = document.getElementById('share-copy-btn');
const shareNativeBtn    = document.getElementById('share-native-btn');
const shareRevokeBtn    = document.getElementById('share-revoke-btn');
const shareCloseBtn     = document.getElementById('share-close-btn');
const shareStatusEl     = document.getElementById('share-status');

function closeShareModal() { shareModalOverlay.classList.remove('show'); }

// One single, permanent link for the whole app/account — not one per chat.
// No login, no server round-trip, no "start a chat first" requirement:
// anyone who opens this URL lands straight on the chat screen and gets
// their own private, anonymous conversation history (see current_username()
// server-side). This is just the site's own root URL.
function openInviteModal() {
  shareLinkInput.value = '';
  shareLinkInput.title = '';
  shareStatusEl.textContent = 'Loading your invite link…';
  shareModalOverlay.classList.add('show');
  shareBtn.classList.add('active');
  if (shareRevokeBtn) shareRevokeBtn.style.display = 'none';  // nothing to revoke — it's a static link
  fetch('/api/invite-link').then(r => r.json()).then(d => {
    const link = d.invite_url || (location.origin + '/');
    shareLinkInput.value = link;
    shareLinkInput.title = link;
    shareStatusEl.textContent = 'Anyone who opens this link can chat with Mythic AI right away — ' +
      'no login needed. Each person gets their own private conversation history; nobody sees yours.';
    requestAnimationFrame(() => { shareLinkInput.focus(); shareLinkInput.select(); });
    renderInviteQrCode(link);
  }).catch(() => {
    shareLinkInput.value = location.origin + '/';
    shareStatusEl.textContent = 'Could not generate a custom link, showing the site link instead.';
    renderInviteQrCode(location.origin + '/');
  });
}

// High error-correction (level H) tolerates ~30% obscured area, which is
// what makes the logo sit safely in the middle without breaking the scan.
function renderInviteQrCode(link) {
  const box = document.getElementById('share-qr-canvas');
  if (!box || typeof QRCode === 'undefined') return;
  box.innerHTML = '';
  new QRCode(box, {
    text: link,
    width: 196,
    height: 196,
    colorDark: '#000000',
    colorLight: '#ffffff',
    correctLevel: QRCode.CorrectLevel.H,
  });
}

if (shareBtn) shareBtn.addEventListener('click', openInviteModal);
if (shareCloseBtn) shareCloseBtn.addEventListener('click', closeShareModal);
if (shareModalOverlay) shareModalOverlay.addEventListener('click', e => { if (e.target === shareModalOverlay) closeShareModal(); });
if (shareLinkInput) shareLinkInput.addEventListener('click', () => shareLinkInput.select());
if (shareOpenBtn) shareOpenBtn.addEventListener('click', () => {
  if (shareLinkInput.value) window.open(shareLinkInput.value, '_blank', 'noopener');
});

if (shareCopyBtn) shareCopyBtn.addEventListener('click', async () => {
  if (!shareLinkInput.value) return;
  try {
    await navigator.clipboard.writeText(shareLinkInput.value);
  } catch {
    shareLinkInput.select();
    try { document.execCommand('copy'); } catch {}
  }
  const orig = shareCopyBtn.textContent;
  shareCopyBtn.textContent = '✓ Copied';
  setTimeout(() => { shareCopyBtn.textContent = orig; }, 1400);
});

if (shareNativeBtn) shareNativeBtn.addEventListener('click', async () => {
  if (!shareLinkInput.value) return;
  if (navigator.share) {
    try { await navigator.share({ title: 'Mythic AI chat', url: shareLinkInput.value }); return; }
    catch (e) { if (e && e.name === 'AbortError') return; }
  }
  shareCopyBtn.click();
});

// shareRevokeBtn is hidden in openInviteModal() — the invite link is static
// and can't be "revoked" (it's just the site's own address). Kept wired to
// closeShareModal() only in case older cached HTML still shows the button.
if (shareRevokeBtn) shareRevokeBtn.addEventListener('click', closeShareModal);

const settingsBtn        = document.getElementById('settings-btn');
const settingsModalOverlay=document.getElementById('settings-modal-overlay');
const settingsCloseBtn   = document.getElementById('settings-close-btn');
const accentColorInput   = document.getElementById('accent-color-input');
const fontSizeSlider     = document.getElementById('font-size-slider');
const fontSizeLabel      = document.getElementById('font-size-label');
const toneSelect         = document.getElementById('tone-select');
const lengthSelect       = document.getElementById('length-select');
const customInstructions = document.getElementById('custom-instructions-input');

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

// ─── API key management (owner only — settings panel) ───────────────────────
const apiKeyLabelInput = document.getElementById('api-key-label-input');
const apiKeyCreateBtn  = document.getElementById('api-key-create-btn');
const apiKeyNewBox     = document.getElementById('api-key-new-box');
const apiKeyNewValue   = document.getElementById('api-key-new-value');
const apiKeyListEl     = document.getElementById('api-key-list');

async function loadApiKeys() {
  if (!apiKeyListEl) return;
  try {
    const res = await fetch('/api/keys');
    if (res.status === 403) { apiKeyListEl.innerHTML = ''; return; } // not the owner — hide silently
    const data = await res.json();
    const keys = data.keys || [];
    apiKeyListEl.innerHTML = keys.length ? '' : '<div class="hint">No keys yet.</div>';
    keys.forEach(k => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:8px;padding:6px 8px;border:1px solid var(--border);border-radius:6px;font-size:13px;flex-wrap:wrap;';
      const used = k.request_count ? `${k.request_count} calls` : 'unused';
      const labelSpan = document.createElement('span');
      labelSpan.innerHTML = `${k.active ? '🟢' : '⚪'} <code>${k.key_prefix}</code> ${k.label ? '— ' + k.label.replace(/</g,'&lt;') : ''} <span style="opacity:.6;">(${used})</span>`;
      row.appendChild(labelSpan);

      const btnGroup = document.createElement('div');
      btnGroup.style.cssText = 'display:flex;gap:6px;';

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.textContent = '📋 Copy';
      copyBtn.className = 'settings-btn';
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(k.key_prefix || '').then(() => {
          const orig = copyBtn.textContent;
          copyBtn.textContent = '✓ Copied';
          setTimeout(() => { copyBtn.textContent = orig; }, 1200);
        });
      };
      btnGroup.appendChild(copyBtn);

      const renameBtn = document.createElement('button');
      renameBtn.type = 'button';
      renameBtn.textContent = '✎ Rename';
      renameBtn.className = 'settings-btn';
      renameBtn.onclick = async () => {
        const newLabel = prompt('New name for this key:', k.label || '');
        if (newLabel === null) return;
        const trimmed = newLabel.trim();
        if (!trimmed) { alert('Name cannot be empty.'); return; }
        const r = await fetch('/api/keys/' + k.id, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label: trimmed }),
        });
        const d = await r.json();
        if (d.error) { alert(d.error); return; }
        loadApiKeys();
        loadApiUsageSummary();
      };
      btnGroup.appendChild(renameBtn);

      if (k.active) {
        const revokeBtn = document.createElement('button');
        revokeBtn.type = 'button';
        revokeBtn.textContent = 'Revoke';
        revokeBtn.className = 'settings-btn';
        revokeBtn.onclick = async () => {
          if (!confirm('Revoke this key? Apps using it will stop working immediately.')) return;
          await fetch('/api/keys/' + k.id, { method: 'DELETE' });
          loadApiKeys();
          loadApiUsageSummary();
        };
        btnGroup.appendChild(revokeBtn);
      }
      row.appendChild(btnGroup);
      apiKeyListEl.appendChild(row);
    });
  } catch (e) {
    console.warn('Could not load API keys:', e);
  }
}

// ─── API usage summary (shown right in the sidebar, under New Chat) ─────────
const apiUsageSummaryEl = document.getElementById('api-usage-summary');

async function loadApiUsageSummary() {
  if (!apiUsageSummaryEl) return;
  try {
    const res = await fetch('/api/keys');
    if (res.status === 403) { apiUsageSummaryEl.style.display = 'none'; return; } // not the owner
    const data = await res.json();
    const keys = data.keys || [];
    if (!keys.length) { apiUsageSummaryEl.style.display = 'none'; return; }

    const activeKeys = keys.filter(k => k.active);
    const totalCalls = keys.reduce((sum, k) => sum + (k.request_count || 0), 0);
    const lastUsedTimes = keys.map(k => k.last_used_at).filter(Boolean).sort();
    const lastUsed = lastUsedTimes.length ? lastUsedTimes[lastUsedTimes.length - 1] : null;
    const lastUsedStr = lastUsed
      ? new Date(lastUsed).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
      : 'never';

    apiUsageSummaryEl.style.display = 'block';
    apiUsageSummaryEl.innerHTML =
      `<div style="display:flex;justify-content:space-between;"><span>🔑 ${activeKeys.length} active key${activeKeys.length === 1 ? '' : 's'}</span>` +
      `<span>${totalCalls} call${totalCalls === 1 ? '' : 's'}</span></div>` +
      `<div style="opacity:.75;margin-top:2px;">Last used: ${lastUsedStr}</div>`;
  } catch (e) {
    console.warn('Could not load API usage summary:', e);
  }
}

if (apiUsageSummaryEl) {
  apiUsageSummaryEl.addEventListener('click', () => {
    openApiUsageOverlay();
  });
  loadApiUsageSummary();
}

// ─── API usage full dashboard (in-page overlay, no page navigation) ─────────
const apiUsageOverlayEl = document.getElementById('api-usage-overlay');
const apiUsageTotalsEl  = document.getElementById('api-usage-totals');
const apiUsageTbodyEl   = document.getElementById('api-usage-tbody');
const apiUsageEmptyEl   = document.getElementById('api-usage-empty');
const apiUsageCloseBtn  = document.getElementById('api-usage-close-btn');
const apiUsageGenBtn    = document.getElementById('api-usage-gen-btn');
const apiUsageCreateOverlayEl = document.getElementById('api-usage-create-overlay');
const apiUsageCreateLabelEl   = document.getElementById('api-usage-create-label');
const apiUsageNewKeyResultEl  = document.getElementById('api-usage-new-key-result');
const apiUsageCreateCloseBtn   = document.getElementById('api-usage-create-close-btn');
const apiUsageCreateConfirmBtn = document.getElementById('api-usage-create-confirm-btn');

function fmtApiUsageDate(iso) {
  if (!iso) return 'Never';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' });
}

async function refreshApiUsageOverlay() {
  if (!apiUsageOverlayEl) return;
  try {
    const res = await fetch('/api/keys');
    if (res.status === 403) {
      apiUsageTotalsEl.innerHTML = '';
      apiUsageTbodyEl.innerHTML = '';
      apiUsageEmptyEl.style.display = 'block';
      apiUsageEmptyEl.textContent = 'Only the account owner can view API keys.';
      return;
    }
    const data = await res.json();
    const keys = data.keys || [];
    const activeCount = keys.filter(k => k.active).length;
    const totalCalls = keys.reduce((s, k) => s + (k.request_count || 0), 0);

    apiUsageTotalsEl.innerHTML = `
      <div style="flex:1;min-width:150px;background:#1a1d24;border:1px solid #2a2e37;border-radius:14px;padding:18px;text-align:center;">
        <div style="font-size:40px;font-weight:800;line-height:1.1;">${activeCount}</div>
        <div style="font-size:12px;color:#9a9ea6;margin-top:6px;letter-spacing:.3px;text-transform:uppercase;">Active Keys</div>
      </div>
      <div style="flex:1;min-width:150px;background:#1a1d24;border:1px solid #2a2e37;border-radius:14px;padding:18px;text-align:center;">
        <div style="font-size:40px;font-weight:800;line-height:1.1;">${totalCalls}</div>
        <div style="font-size:12px;color:#9a9ea6;margin-top:6px;letter-spacing:.3px;text-transform:uppercase;">Total Calls</div>
      </div>
      <div style="flex:1;min-width:150px;background:#1a1d24;border:1px solid #2a2e37;border-radius:14px;padding:18px;text-align:center;">
        <div style="font-size:40px;font-weight:800;line-height:1.1;">${keys.length}</div>
        <div style="font-size:12px;color:#9a9ea6;margin-top:6px;letter-spacing:.3px;text-transform:uppercase;">Total Keys</div>
      </div>`;

    if (!keys.length) {
      apiUsageTbodyEl.innerHTML = '';
      apiUsageEmptyEl.style.display = 'block';
      apiUsageEmptyEl.textContent = 'No API keys yet. Click "Generate API Key" to create one.';
      return;
    }
    apiUsageEmptyEl.style.display = 'none';

    apiUsageTbodyEl.innerHTML = keys.map(k => `
      <tr>
        <td style="padding:16px;border-bottom:1px solid #22252c;font-size:14px;font-weight:700;">${(k.label || '(unnamed key)').replace(/</g,'&lt;')}</td>
        <td style="padding:16px;border-bottom:1px solid #22252c;font-size:14px;font-family:monospace;color:#c7cad1;">
          <div style="display:flex;align-items:center;gap:8px;"><span>${k.key_prefix || ''}</span>
          <button class="api-usage-copy-btn" data-prefix="${(k.key_prefix||'').replace(/"/g,'&quot;')}" style="background:none;border:1px solid #3a3e47;color:#c7cad1;cursor:pointer;font-size:11.5px;padding:4px 9px;border-radius:6px;font-weight:600;font-family:inherit;white-space:nowrap;">📋 Copy</button></div>
        </td>
        <td style="padding:16px;border-bottom:1px solid #22252c;font-size:14px;">${fmtApiUsageDate(k.created_at)}</td>
        <td style="padding:16px;border-bottom:1px solid #22252c;font-size:18px;font-weight:700;">${k.request_count || 0}</td>
        <td style="padding:16px;border-bottom:1px solid #22252c;font-size:14px;">
          <span style="font-size:11px;font-weight:700;letter-spacing:.4px;border:1px solid;border-radius:20px;padding:3px 10px;color:${k.active ? '#1a9e5c' : '#c0392b'};border-color:${k.active ? '#1a9e5c' : '#c0392b'};">${k.active ? 'ACTIVE' : 'REVOKED'}</span>
        </td>
        <td style="padding:16px;border-bottom:1px solid #22252c;font-size:14px;">
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <button class="api-usage-rename-btn" data-id="${k.id}" data-label="${(k.label||'').replace(/"/g,'&quot;')}" style="background:none;border:1px solid #3a3e47;color:#9a9ea6;border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer;font-family:inherit;">✎ Rename</button>
            ${k.active ? `<button class="api-usage-revoke-btn" data-id="${k.id}" style="background:none;border:1px solid #3a3e47;color:#c0392b;border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer;font-family:inherit;">Revoke</button>` : ''}
          </div>
        </td>
      </tr>`).join('');

    apiUsageTbodyEl.querySelectorAll('.api-usage-revoke-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (!confirm('Revoke this key? Apps using it will stop working immediately.')) return;
        await fetch('/api/keys/' + btn.dataset.id, { method: 'DELETE' });
        refreshApiUsageOverlay();
        loadApiUsageSummary();
      });
    });
    apiUsageTbodyEl.querySelectorAll('.api-usage-copy-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        navigator.clipboard.writeText(btn.dataset.prefix || '').then(() => {
          const orig = btn.textContent;
          btn.textContent = '✓ Copied';
          setTimeout(() => { btn.textContent = orig; }, 1200);
        });
      });
    });
    apiUsageTbodyEl.querySelectorAll('.api-usage-rename-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const newLabel = prompt('New name for this key:', btn.dataset.label || '');
        if (newLabel === null) return;
        const trimmed = newLabel.trim();
        if (!trimmed) { alert('Name cannot be empty.'); return; }
        const r = await fetch('/api/keys/' + btn.dataset.id, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label: trimmed }),
        });
        const d = await r.json();
        if (d.error) { alert(d.error); return; }
        refreshApiUsageOverlay();
        loadApiUsageSummary();
      });
    });
  } catch (e) {
    console.warn('Could not load API usage:', e);
  }
}

function openApiUsageOverlay() {
  if (!apiUsageOverlayEl) return;
  apiUsageOverlayEl.style.display = 'block';
  refreshApiUsageOverlay();
}
function closeApiUsageOverlay() {
  if (apiUsageOverlayEl) apiUsageOverlayEl.style.display = 'none';
}
if (apiUsageCloseBtn) apiUsageCloseBtn.addEventListener('click', closeApiUsageOverlay);

if (apiUsageGenBtn) apiUsageGenBtn.addEventListener('click', () => {
  apiUsageCreateLabelEl.value = '';
  apiUsageNewKeyResultEl.innerHTML = '';
  apiUsageCreateOverlayEl.style.display = 'flex';
});
if (apiUsageCreateCloseBtn) apiUsageCreateCloseBtn.addEventListener('click', () => {
  apiUsageCreateOverlayEl.style.display = 'none';
  refreshApiUsageOverlay();
  loadApiUsageSummary();
});
if (apiUsageCreateConfirmBtn) apiUsageCreateConfirmBtn.addEventListener('click', async () => {
  const label = apiUsageCreateLabelEl.value.trim();
  apiUsageCreateConfirmBtn.disabled = true;
  try {
    const res = await fetch('/api/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    const data = await res.json();
    if (data.api_key) {
      apiUsageNewKeyResultEl.innerHTML =
        '<div style="background:#0f1115;border:1px solid #1a9e5c;border-radius:8px;padding:12px;margin-bottom:14px;">' +
          '<div style="display:flex;gap:8px;align-items:center;">' +
            '<input type="text" id="api-usage-new-key-input" readonly value="' + data.api_key.replace(/"/g, '&quot;') + '" ' +
              'style="flex:1;background:#0f1115;border:1px solid #1a9e5c;border-radius:6px;padding:8px;font-family:monospace;font-size:12px;color:#7be3ab;box-sizing:border-box;min-width:0;">' +
            '<button type="button" id="api-usage-new-key-copy-btn" ' +
              'style="background:#1a9e5c;color:#04140b;border:none;padding:8px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:700;white-space:nowrap;">Copy</button>' +
          '</div>' +
        '</div>' +
        '<div style="font-size:12px;color:#9a9ea6;margin-bottom:10px;">Copy this now — it will not be shown again. (Double-clicking to select may only grab part of the key due to the hyphens — use the Copy button instead.)</div>';
      const newKeyCopyBtn = document.getElementById('api-usage-new-key-copy-btn');
      if (newKeyCopyBtn) {
        newKeyCopyBtn.addEventListener('click', async () => {
          const keyInput = document.getElementById('api-usage-new-key-input');
          if (!keyInput) return;
          try {
            keyInput.select();
            document.execCommand('copy');
            newKeyCopyBtn.textContent = '✓ Copied!';
            setTimeout(() => { newKeyCopyBtn.textContent = 'Copy'; }, 2500);
          } catch (e) {
            try {
              await navigator.clipboard.writeText(keyInput.value);
              newKeyCopyBtn.textContent = '✓ Copied!';
              setTimeout(() => { newKeyCopyBtn.textContent = 'Copy'; }, 2500);
            } catch (e2) {
              alert('Copy failed. Try selecting the whole box manually (click once, then Ctrl+A, Ctrl+C).');
            }
          }
        });
      }
      refreshApiUsageOverlay();
      loadApiUsageSummary();
    } else if (data.error) {
      alert(data.error);
    }
  } catch (e) {
    alert('Could not create key.');
  } finally {
    apiUsageCreateConfirmBtn.disabled = false;
  }
});

if (apiKeyCreateBtn) {
  apiKeyCreateBtn.addEventListener('click', apiKeyCreateHandler);
  // iOS fallback: add touchstart for better compatibility
  if (isIOS) {
    apiKeyCreateBtn.addEventListener('touchstart', apiKeyCreateHandler, { passive: false });
  }
  
  async function apiKeyCreateHandler(e) {
    const label = apiKeyLabelInput ? apiKeyLabelInput.value.trim() : '';
    apiKeyCreateBtn.disabled = true;
    try {
      const res = await fetch('/api/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      
      const data = await res.json();
      
      if (data.error) {
        alert('Error: ' + (data.error.message || data.error));
        return;
      }
      
      if (data.api_key) {
        apiKeyNewValue.value = data.api_key;
        apiKeyNewBox.style.display = 'block';
        if (apiKeyLabelInput) apiKeyLabelInput.value = '';
        loadApiKeys();
        loadApiUsageSummary();
      } else {
        alert('No API key returned from server');
      }
    } catch (e) {
      console.error('[API Key Creation Error]', e);
      alert('Could not create API key: ' + e.message);
    } finally {
      apiKeyCreateBtn.disabled = false;
    }
  }
}

const apiKeyCopyBtn = document.getElementById('api-key-copy-btn');
if (apiKeyCopyBtn) {
  apiKeyCopyBtn.addEventListener('click', async () => {
    const keyInput = document.getElementById('api-key-new-value');
    const keyValue = keyInput ? keyInput.value : '';
    console.log('=== API Key Copy Debug ===');
    console.log('keyInput element:', keyInput);
    console.log('keyInput.value length:', keyValue.length);
    console.log('keyInput.value (first 50 chars):', keyValue.substring(0, 50));
    console.log('Full keyInput.value:', keyValue);
    
    if (!keyValue || keyValue.length < 20) {
      alert(`Key is incomplete (only ${keyValue.length} chars). Check browser console for details.`);
      return;
    }
    
    try {
      keyInput.select();
      document.execCommand('copy');
      apiKeyCopyBtn.textContent = '✓ Copied!';
      console.log('✓ Copy succeeded');
      setTimeout(() => { apiKeyCopyBtn.textContent = 'Copy'; }, 2500);
    } catch (e) {
      console.error('execCommand failed:', e);
      try {
        await navigator.clipboard.writeText(keyValue);
        apiKeyCopyBtn.textContent = '✓ Copied!';
        console.log('✓ Clipboard API succeeded');
        setTimeout(() => { apiKeyCopyBtn.textContent = 'Copy'; }, 2500);
      } catch (e2) {
        console.error('Both methods failed:', e2);
        alert('Copy failed. Try Ctrl+A then Ctrl+C in the box above.');
      }
    }
  });
}

function applyTheme(t) {
  if (t === 'system') {
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.body.classList.toggle('theme-light', !dark);
  } else {
    document.body.classList.toggle('theme-light', t === 'light');
  }
}

settingsBtn.addEventListener('click', () => { settingsModalOverlay.style.display = 'flex'; loadApiKeys(); });

const apiKeysShortcutBtn = document.getElementById('api-keys-shortcut-btn');
if (apiKeysShortcutBtn) apiKeysShortcutBtn.addEventListener('click', () => {
  openApiUsageOverlay();
});
settingsCloseBtn.addEventListener('click', () => { saveSettings(); settingsModalOverlay.style.display = 'none'; });
settingsModalOverlay.addEventListener('click', e => { if (e.target === settingsModalOverlay) { saveSettings(); settingsModalOverlay.style.display = 'none'; } });

(function() {
  const notifBtn    = document.getElementById('notif-toggle-btn');
  const notifStatus = document.getElementById('notif-status');
  if (!notifBtn) return;

  function updateNotifUI() {
    if (isIOS) {
      notifBtn.textContent = 'Not available on iPhone';
      notifBtn.disabled = true;
      notifStatus.textContent = 'Push notifications are not supported in Safari PWA. App works normally without them.';
      return;
    }
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

// Detect if running on iPhone/iOS
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;

if ('serviceWorker' in navigator && !isIOS) {
  // Service workers are unreliable on iOS, skip on iPhone
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
} else if (isIOS) {
  // iPhone: skip push notifications (not supported in Safari PWA)
  // App still works fine, just no push notifications
  console.log('[iOS] Push notifications not supported in Safari PWA');
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
  await loadConversationList();

  // If we were redirected here from a "Continue this conversation" share
  // link (see SHARE_PAGE), open that freshly-forked conversation instead
  // of the usual blank New Chat screen. Also support a plain "?c=<id>"
  // link/refresh/bookmark — this is the id openConversation() itself now
  // puts in the address bar, so refreshing or sharing that URL reopens
  // the same chat instead of always landing on a blank screen.
  const params = new URLSearchParams(location.search);
  const openId = params.get('open') || params.get('c');
  if (openId) {
    // Normalize the URL to ?c=<id> and open without pushing a duplicate
    // history entry (replaceState keeps Back from bouncing between the
    // raw ?open= link and the normalized ?c= one).
    history.replaceState({ conv: openId }, '', '?c=' + encodeURIComponent(openId));
    await openConversation(openId, { updateUrl: false });
  } else {
    // Always start on a fresh New Chat screen otherwise, rather than
    // auto-reopening the last conversation — the sidebar list is still
    // populated underneath.
    showEmptyState();
  }

  // Silently remove stray internal-tooling conversations (old follow-up-
  // suggestion / instruction-prefix leaks) in the background, no button
  // or confirmation needed — safe because it only ever matches those very
  // specific known patterns server-side (see _JUNK_CONV_PATTERNS).
  try {
    const r = await fetch('/api/conversations/cleanup-junk', { method: 'POST' });
    const d = await r.json();
    if (d && d.removed_count > 0) loadConversationList();
  } catch {}
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
  const hwModal = document.getElementById('homework-modal-overlay');
  if (hwModal) { hwModal.style.display = 'flex'; document.getElementById('homework-question').focus(); }
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

// ─── HOMEWORK & STUDY BOOK MODAL ─────────────────────────────────────────────
(function() {
  const modal       = document.getElementById('homework-modal-overlay');
  const closeBtn     = document.getElementById('hw-close-btn');
  const sendBtn      = document.getElementById('hw-send-btn');
  const questionEl   = document.getElementById('homework-question');
  const modeUploadBtn= document.getElementById('hw-mode-upload');
  const modeUrlBtn   = document.getElementById('hw-mode-url');
  const uploadArea   = document.getElementById('hw-upload-area');
  const urlArea      = document.getElementById('hw-url-area');
  const dropzone     = document.getElementById('hw-upload-dropzone');
  const hwFileInput  = document.getElementById('hw-file-input');
  const hwFileName   = document.getElementById('hw-file-name');
  const urlInput     = document.getElementById('hw-url-input');
  const loadingEl    = document.getElementById('hw-loading');
  const errorEl      = document.getElementById('hw-error');
  if (!modal) return;

  let hwMode = 'upload';
  let hwPendingFile = null; // { name, mimeType, dataBase64 }

  function setHwMode(mode) {
    hwMode = mode;
    modeUploadBtn.style.borderColor = mode === 'upload' ? 'var(--accent)' : 'var(--border)';
    modeUploadBtn.style.background  = mode === 'upload' ? 'var(--accent-dim)' : 'var(--panel)';
    modeUploadBtn.style.color       = mode === 'upload' ? 'var(--accent)' : 'var(--muted)';
    modeUrlBtn.style.borderColor    = mode === 'url' ? 'var(--accent)' : 'var(--border)';
    modeUrlBtn.style.background     = mode === 'url' ? 'var(--accent-dim)' : 'var(--panel)';
    modeUrlBtn.style.color          = mode === 'url' ? 'var(--accent)' : 'var(--muted)';
    uploadArea.style.display = mode === 'upload' ? 'block' : 'none';
    urlArea.style.display    = mode === 'url' ? 'block' : 'none';
  }
  modeUploadBtn.addEventListener('click', () => setHwMode('upload'));
  modeUrlBtn.addEventListener('click', () => setHwMode('url'));

  dropzone.addEventListener('click', () => hwFileInput.click());
  hwFileInput.addEventListener('change', () => {
    const file = hwFileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      hwPendingFile = { name: file.name, mimeType: file.type || 'application/octet-stream', dataBase64: e.target.result.split(',')[1] };
      hwFileName.textContent = '📄 ' + file.name;
      hwFileName.style.display = 'block';
    };
    reader.readAsDataURL(file);
  });

  if (homeworkBtn) homeworkBtn.addEventListener('click', () => {
    errorEl.style.display = 'none';
    loadingEl.style.display = 'none';
  });
  if (closeBtn) closeBtn.addEventListener('click', () => { modal.style.display = 'none'; });
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.style.display = 'none'; });

  const STUDY_INSTRUCTIONS = "First list the chapters you can see in this document and tell me which chapter you're focusing on. Then prepare the hardest, most exam-relevant questions for that chapter first, before easier ones.";

  sendBtn.addEventListener('click', async () => {
    errorEl.style.display = 'none';
    const question = questionEl.value.trim();

    if (hwMode === 'upload') {
      if (!hwPendingFile && !question) {
        errorEl.textContent = 'Upload a file or type a question first.';
        errorEl.style.display = 'block';
        return;
      }
      const finalMessage = question || STUDY_INSTRUCTIONS;
      modal.style.display = 'none';
      addMessage('user', finalMessage, hwPendingFile);
      const attachmentToSend = hwPendingFile;
      hwPendingFile = null;
      hwFileName.style.display = 'none';
      hwFileInput.value = '';
      questionEl.value = '';
      streamReply({ message: getTonePrefix() + finalMessage, attachment: attachmentToSend });
      return;
    }

    // URL mode — fetch + extract text server-side, then send as a normal
    // text message with the extracted content inlined (same 12k-char cap
    // the file-upload path already uses via extract_text_from_attachment).
    const url = urlInput.value.trim();
    if (!url && !question) {
      errorEl.textContent = 'Paste a URL or type a question first.';
      errorEl.style.display = 'block';
      return;
    }
    if (!url) {
      // No URL, just a plain question — send as normal chat.
      modal.style.display = 'none';
      addMessage('user', question);
      questionEl.value = '';
      streamReply({ message: getTonePrefix() + question });
      return;
    }

    loadingEl.style.display = 'block';
    sendBtn.disabled = true;
    try {
      const r = await fetch('/api/fetch-url-document', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      const d = await r.json();
      loadingEl.style.display = 'none';
      if (!r.ok || d.error) {
        errorEl.textContent = d.error || 'Could not fetch that URL.';
        errorEl.style.display = 'block';
        return;
      }
      const finalMessage = (question || STUDY_INSTRUCTIONS)
        + `\n\n[Document fetched from ${d.filename || url}]\n${d.text}`
        + (d.note ? `\n[Note: ${d.note}]` : '');
      modal.style.display = 'none';
      addMessage('user', question || `📚 Studying: ${d.filename || url}`);
      urlInput.value = '';
      questionEl.value = '';
      streamReply({ message: getTonePrefix() + finalMessage });
    } catch (e) {
      loadingEl.style.display = 'none';
      errorEl.textContent = 'Network error: ' + e.message;
      errorEl.style.display = 'block';
    } finally {
      sendBtn.disabled = false;
    }
  });
})();

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
    const canRunCode = !canPreview && RUNNABLE_LANGS.includes((a.lang || '').toLowerCase());
    return `
    <div class="art-row" data-id="${a.id}" style="border:1px solid var(--border);border-radius:10px;margin-bottom:8px;overflow:hidden;">
      <div style="display:flex;justify-content:space-between;align-items:center;background:var(--panel);padding:8px 12px;font-size:11.5px;color:var(--muted);">
        <span>📦 ${a.lang} &middot; ${a.preview || 'snippet'}</span>
        <div style="display:flex;gap:6px;">
          ${canPreview ? `<button class="art-run" data-group="${a.groupId}" style="background:var(--accent);border:none;color:#fff;border-radius:6px;cursor:pointer;font-size:11px;padding:3px 8px;">▶ Run</button>` : ''}
          ${canRunCode ? `<button class="art-run-code" data-id="${a.id}" style="background:var(--accent);border:none;color:#fff;border-radius:6px;cursor:pointer;font-size:11px;padding:3px 8px;">▶ Run</button>` : ''}
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

// ─── Real code execution ("Run") for non-HTML artifacts, e.g. Python ───────
// HTML/CSS/JS artifacts already run for real, in-browser, via the sandboxed
// iframe above. Other languages (Python, etc.) can't run in a browser at
// all, so this calls the server's /api/execute-code, which itself runs the
// code on Piston (a third-party sandbox) rather than on this app's own
// server — see the long comment above api_execute_code() in the Python for
// why that separation matters. Gated behind VIP the same way as the rest of
// Cowork/Code/Artifacts.
const RUNNABLE_LANGS = ['python', 'py', 'javascript', 'js', 'bash', 'sh', 'c', 'cpp', 'c++', 'java', 'go', 'rust', 'ruby', 'php'];
function showCodeRunResultModal(title, body) {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:650;display:flex;align-items:center;justify-content:center;padding:20px;';
  overlay.innerHTML = `<div style="background:#111;border:1px solid #333;border-radius:12px;padding:16px;width:92%;max-width:640px;max-height:78vh;overflow-y:auto;color:#eee;font-family:ui-monospace,monospace;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <strong style="font-size:13px;">▶ ${title}</strong>
      <button id="run-result-close" style="background:none;border:1px solid #444;color:#ccc;border-radius:6px;padding:4px 10px;cursor:pointer;font-family:inherit;">✕</button>
    </div>
    <pre style="white-space:pre-wrap;font-size:12.5px;margin:0;">${body}</pre>
  </div>`;
  document.body.appendChild(overlay);
  overlay.querySelector('#run-result-close').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}
async function runArtifactCode(artifactId) {
  const a = _artifacts.find(x => x.id === artifactId);
  if (!a) return;
  const lang = (a.lang || '').toLowerCase();
  if (!RUNNABLE_LANGS.includes(lang)) {
    showCodeRunResultModal('Can\'t run this', `"${lang}" isn't a runnable language here.`);
    return;
  }
  showCodeRunResultModal('Running…', 'Executing on a sandboxed runner, one moment…');
  try {
    const r = await fetch('/api/execute-code', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: a.code, language: lang }),
    });
    const d = await r.json();
    if (!r.ok || d.error) {
      showCodeRunResultModal('Run failed', (d.error || 'Unknown error') +
        (r.status === 403 ? '\n\nUnlock VIP mode first (🔒 button in the header).' : ''));
      return;
    }
    const out = [
      d.stdout ? 'stdout:\n' + d.stdout : '',
      d.stderr ? 'stderr:\n' + d.stderr : '',
      (!d.stdout && !d.stderr) ? '(no output)' : '',
      `\n[exit code ${d.exit_code}]`,
    ].filter(Boolean).join('\n\n');
    showCodeRunResultModal(`Result (${d.language} ${d.version || ''})`, out);
  } catch (err) {
    showCodeRunResultModal('Network error', String(err.message || err));
  }
}
// Hook into the Artifacts modal's row buttons for non-HTML runnable code —
// showArtifactsModal() only wires up '.art-run' for HTML groups today, so
// this listens at the document level and covers any '.art-run-code' button
// we add per-row (see the small patch to the row template just below).
document.addEventListener('click', e => {
  const btn = e.target.closest('.art-run-code');
  if (btn) runArtifactCode(btn.dataset.id);
});

// ─── Message search across every chat (not just the open one) ─────────────
const searchChatsBtn = document.getElementById('search-chats-btn');
function showSearchModal() {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;padding:20px;';
  overlay.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px;width:92%;max-width:520px;max-height:78vh;overflow-y:auto;">
    <h3 style="margin:0 0 12px;font-size:16px;">🔎 Search your chats</h3>
    <input id="search-chats-input" type="text" placeholder="Search all conversations..." autocomplete="off"
      style="width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-family:inherit;font-size:14px;margin-bottom:12px;">
    <div id="search-chats-results" style="font-size:13px;color:var(--muted);"></div>
    <button id="search-chats-close" style="margin-top:10px;width:100%;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:9px;cursor:pointer;font-family:inherit;">Close</button>
  </div>`;
  document.body.appendChild(overlay);
  const input2 = overlay.querySelector('#search-chats-input');
  const resultsEl = overlay.querySelector('#search-chats-results');
  let debounceTimer = null;
  input2.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    const q = input2.value.trim();
    if (q.length < 2) { resultsEl.innerHTML = ''; return; }
    debounceTimer = setTimeout(async () => {
      resultsEl.textContent = 'Searching…';
      try {
        const r = await fetch('/api/search?q=' + encodeURIComponent(q));
        const d = await r.json();
        const results = d.results || [];
        resultsEl.innerHTML = results.length ? results.map(res => `
          <div class="search-result-row" data-conv="${res.conv_id}" style="padding:10px;border:1px solid var(--border);border-radius:8px;margin-bottom:6px;cursor:pointer;">
            <div style="font-weight:600;font-size:12.5px;margin-bottom:3px;">${(res.title || '').replace(/</g,'&lt;')}</div>
            <div style="opacity:.8;">${(res.role === 'user' ? '🧑' : '🤖')} ${(res.snippet || '').replace(/</g,'&lt;')}</div>
          </div>`).join('') : '<div style="padding:12px;text-align:center;">No matches.</div>';
        resultsEl.querySelectorAll('.search-result-row').forEach(row => {
          row.addEventListener('click', () => { overlay.remove(); openConversation(row.dataset.conv); });
        });
      } catch { resultsEl.textContent = 'Search failed — try again.'; }
    }, 300);
  });
  overlay.querySelector('#search-chats-close').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  requestAnimationFrame(() => input2.focus());
}
if (searchChatsBtn) searchChatsBtn.addEventListener('click', showSearchModal);

// ─── Reminders & scheduled tasks (delivered via existing Web Push) ─────────
const remindersBtn = document.getElementById('reminders-btn');
async function loadRemindersList(container) {
  container.textContent = 'Loading…';
  try {
    const r = await fetch('/api/reminders');
    const d = await r.json();
    const list = d.reminders || [];
    container.innerHTML = list.length ? list.map(rem => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 10px;border:1px solid var(--border);border-radius:8px;margin-bottom:6px;font-size:12.5px;">
        <div>
          <div>${rem.text.replace(/</g,'&lt;')}</div>
          <div style="opacity:.7;font-size:11px;">${new Date(rem.fire_at * 1000).toLocaleString()}</div>
        </div>
        <button class="reminder-del" data-id="${rem.id}" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:14px;">✕</button>
      </div>`).join('') : '<div style="padding:12px;text-align:center;color:var(--muted);">No reminders set.</div>';
    container.querySelectorAll('.reminder-del').forEach(btn => btn.addEventListener('click', async () => {
      await fetch('/api/reminders/' + btn.dataset.id, { method: 'DELETE' });
      loadRemindersList(container);
    }));
  } catch { container.textContent = 'Could not load reminders.'; }
}
function showRemindersModal() {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:500;display:flex;align-items:center;justify-content:center;padding:20px;';
  const now = new Date(Date.now() + 5 * 60000);
  const defaultLocal = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  overlay.innerHTML = `<div style="background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px;width:92%;max-width:480px;max-height:80vh;overflow-y:auto;">
    <h3 style="margin:0 0 12px;font-size:16px;">⏰ Reminders</h3>
    <p style="font-size:12px;color:var(--muted);margin:0 0 12px;">Delivered as a push notification (enable notifications in Settings first). Requires the app to stay running on the server (works on Render; not on serverless hosts like Vercel).</p>
    <input id="reminder-text-input" type="text" placeholder="Remind me to..." maxlength="200"
      style="width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-family:inherit;font-size:14px;margin-bottom:8px;">
    <input id="reminder-time-input" type="datetime-local" value="${defaultLocal}"
      style="width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-family:inherit;font-size:14px;margin-bottom:10px;">
    <button id="reminder-add-btn" style="width:100%;background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px;cursor:pointer;font-family:inherit;margin-bottom:14px;">Set reminder</button>
    <div id="reminders-list"></div>
    <button id="reminders-close" style="margin-top:10px;width:100%;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:9px;cursor:pointer;font-family:inherit;">Close</button>
  </div>`;
  document.body.appendChild(overlay);
  const listEl = overlay.querySelector('#reminders-list');
  loadRemindersList(listEl);
  overlay.querySelector('#reminder-add-btn').addEventListener('click', async () => {
    const text = overlay.querySelector('#reminder-text-input').value.trim();
    const localVal = overlay.querySelector('#reminder-time-input').value;
    if (!text || !localVal) return;
    const fireAt = new Date(localVal).getTime() / 1000;
    try {
      const r = await fetch('/api/reminders', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, fire_at: fireAt }),
      });
      const d = await r.json();
      if (!r.ok) { alert(d.error || 'Could not set reminder.'); return; }
      overlay.querySelector('#reminder-text-input').value = '';
      loadRemindersList(listEl);
    } catch (err) { alert('Network error: ' + err.message); }
  });
  overlay.querySelector('#reminders-close').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}
if (remindersBtn) remindersBtn.addEventListener('click', showRemindersModal);

// ─── Full backup export / import (all chats, not just one) ────────────────
// Wired into the Settings modal — see #backup-export-btn / #backup-import-*
// elements added to the settings modal markup.
const backupExportBtn = document.getElementById('backup-export-btn');
const backupImportBtn = document.getElementById('backup-import-btn');
const backupImportFile = document.getElementById('backup-import-file');
const backupStatusEl   = document.getElementById('backup-status');
if (backupExportBtn) backupExportBtn.addEventListener('click', () => {
  window.location.href = '/api/backup/export';
});
if (backupImportBtn && backupImportFile) {
  backupImportBtn.addEventListener('click', () => backupImportFile.click());
  backupImportFile.addEventListener('change', async () => {
    const file = backupImportFile.files[0];
    if (!file) return;
    if (backupStatusEl) backupStatusEl.textContent = 'Importing…';
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('/api/backup/import', { method: 'POST', body: fd });
      const d = await r.json();
      if (backupStatusEl) {
        backupStatusEl.textContent = r.ok
          ? `Imported ${d.imported} conversation(s).`
          : (d.error || 'Import failed.');
      }
      loadConversationList();
    } catch (err) {
      if (backupStatusEl) backupStatusEl.textContent = 'Network error: ' + err.message;
    }
    backupImportFile.value = '';
  });
}

// ─── Cowork mode: real multi-step task runner ──────────────────────────────
// Renders a distinct "steps" card (plan → each step's result → final
// answer) instead of a normal streamed reply, so it's visibly different
// from a single chat completion.
function renderCoworkResult(container, data) {
  const stepsHtml = (data.steps || []).map((s, i) => `
    <div style="margin-bottom:8px;padding:8px 10px;border-left:2px solid var(--accent);">
      <div style="font-size:11.5px;font-weight:600;opacity:.8;">Step ${i + 1}: ${s.step.replace(/</g,'&lt;')}</div>
      <div style="font-size:13px;margin-top:3px;">${s.result.replace(/</g,'&lt;')}</div>
    </div>`).join('');
  container.innerHTML = `
    <div style="font-size:12px;font-weight:700;opacity:.7;margin-bottom:8px;">🗂 COWORK — ${(data.steps||[]).length} step(s)</div>
    ${stepsHtml}
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);white-space:pre-wrap;">${(data.final_answer || '').replace(/</g,'&lt;')}</div>`;
}
async function runCoworkTask(task) {
  addMessage('user', task);
  // addMessage() returns the message's text element directly (see its
  // definition above), not the whole row — safe to update in place.
  const textEl = addMessage('ai', '⏳ Planning and working through the task…');
  try {
    const r = await fetch('/api/cowork/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task }),
    });
    const d = await r.json();
    if (!r.ok || d.error) {
      if (textEl) textEl.textContent = d.error || 'Cowork task failed.';
      return;
    }
    if (textEl) renderCoworkResult(textEl, d);
  } catch (err) {
    if (textEl) textEl.textContent = 'Network error: ' + err.message;
  }
}

// ─── Starred view toggle ─────────────────────────────────────────────────────
const archivedToggleBtn = document.getElementById('archived-toggle-btn');
if (archivedToggleBtn) archivedToggleBtn.addEventListener('click', () => requirePassword(() => {
  showingStarredOnly = !showingStarredOnly;
  archivedToggleBtn.textContent = showingStarredOnly ? '💬 All Chats' : '⭐ Starred';
  archivedToggleBtn.style.color = showingStarredOnly ? 'var(--accent)' : '';
  archivedToggleBtn.style.borderColor = showingStarredOnly ? 'var(--accent)' : 'var(--border)';
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
    <h3 style="margin:0 0 12px;font-size:16px;">🔖 Bookmarked Messages</h3>
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
  bmBtn.textContent = isBookmarked(activeConvId, myIndex) ? '🔖' : '📑';
  bmBtn.addEventListener('click', () => {
    const on = toggleBookmark(activeConvId, myIndex, textNode.textContent || textNode.innerText || '');
    bmBtn.textContent = on ? '🔖' : '📑';
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
    { label: '🔖 Bookmarked Messages', action: showBookmarksModal },
    { label: '⭐ Toggle Starred View', action: () => archivedToggleBtn && archivedToggleBtn.click() },
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

# ── Public, read-only "shared chat" page ──────────────────────────────────
# Deliberately a separate, minimal template: no login, no sidebar, no
# composer, no access to the viewer's own conversations — it only ever
# fetches /api/share/<id>, which itself only ever returns the one shared
# conversation's messages.
SHARE_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex">
<link rel="icon" type="image/png" href="/icon.png">
<title>Shared chat · Mythic AI</title>
<style>
  :root { --bg:#1a1a1a; --panel:#2a2a2a; --border:#3a3a3a; --text:#ececec;
    --muted:#8e8ea0; --accent:#10a37f; --user-bubble:#2a2a2a; --ai-bubble:#1a1a1a; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,
    "Segoe UI",Inter,sans-serif; min-height:100vh; }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--border);
    padding:14px 20px; display:flex; align-items:center; justify-content:space-between; gap:10px;
    z-index:10; }
  header .brand { display:flex; align-items:center; gap:8px; font-weight:700; color:var(--accent);
    font-variant:small-caps; letter-spacing:.5px; }
  header .brand img { width:26px; height:26px; border-radius:7px; display:block; }
  header .badge { display:inline-flex; align-items:center; gap:6px; font-size:11px; color:var(--muted);
    border:1px solid var(--border); border-radius:20px; padding:5px 12px; }
  header .badge svg { width:13px; height:13px; flex-shrink:0; }
  header a.cta { background:var(--accent); color:#fff; text-decoration:none; font-size:12.5px;
    font-weight:700; padding:8px 14px; border-radius:8px; }
  #wrap { max-width:760px; margin:0 auto; padding:20px; }
  #title { font-size:19px; font-weight:700; margin-bottom:18px; }
  .msg-row { display:flex; flex-direction:column; margin-bottom:14px; max-width:82%; }
  .msg-row.user { margin-left:auto; align-items:flex-end; }
  .msg-row.ai { margin-right:auto; align-items:flex-start; }
  .msg { padding:11px 15px; border-radius:18px; line-height:1.6; font-size:14.5px;
    white-space:pre-wrap; word-wrap:break-word; }
  .msg-row.user .msg { background:var(--user-bubble); border-bottom-right-radius:4px; }
  .msg-row.ai .msg { background:var(--ai-bubble); border-bottom-left-radius:4px; }
  #state { text-align:center; color:var(--muted); padding:60px 20px; font-size:14px; }
  #footer-cta { text-align:center; padding:14px 20px 40px; color:var(--muted); font-size:13px; }
  #footer-cta a { color:var(--accent); text-decoration:none; font-weight:600; }
  code { background:var(--panel); padding:1px 5px; border-radius:4px; font-size:.92em; }
  pre { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 12px;
    overflow-x:auto; margin:8px 0; }
  #continue-bar { max-width:760px; margin:0 auto; padding:0 20px 20px; }
  #continue-btn { display:flex; align-items:center; justify-content:center; gap:8px; width:100%;
    background:var(--accent); color:#fff; border:none; border-radius:12px; padding:13px;
    font-size:14px; font-weight:700; cursor:pointer; font-family:inherit; }
  #continue-btn:hover { opacity:.92; }
  #continue-btn:disabled { opacity:.6; cursor:default; }
  #continue-error { max-width:760px; margin:0 auto; padding:0 20px 12px; color:#ef4444; font-size:12.5px; display:none; }
</style>
</head>
<body>
<header>
  <span class="brand"><img src="/icon.png" alt="Mythic AI"> Mythic AI</span>
  <span class="badge">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
    </svg>
    Shared chat · read-only
  </span>
  <a class="cta" href="/">Open Mythic AI</a>
</header>
<div id="wrap">
  <div id="title"></div>
  <div id="messages"></div>
  <div id="state" style="display:none;"></div>
</div>
<div id="continue-error"></div>
<div id="continue-bar" style="display:none;">
  <button id="continue-btn" type="button">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
    Continue this conversation
  </button>
</div>
<div id="footer-cta">Want a conversation like this? <a href="/">Try Mythic AI</a></div>
<script>
function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function renderInline(text) {
  let html = esc(text);
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code>${code.trim()}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\n/g, '<br>');
  return html;
}
const shareId = location.pathname.split('/').filter(Boolean).pop();
(async () => {
  const stateEl = document.getElementById('state');
  try {
    const r = await fetch('/api/share/' + encodeURIComponent(shareId));
    const d = await r.json();
    if (!r.ok || d.error) {
      stateEl.textContent = d.error || 'This shared chat could not be found.';
      stateEl.style.display = 'block';
      return;
    }
    document.title = (d.title || 'Shared chat') + ' · Mythic AI';
    document.getElementById('title').textContent = d.title || 'Shared chat';
    const wrap = document.getElementById('messages');
    (d.messages || []).forEach(m => {
      const row = document.createElement('div');
      row.className = 'msg-row ' + (m.role === 'user' ? 'user' : 'ai');
      const bubble = document.createElement('div');
      bubble.className = 'msg';
      bubble.innerHTML = renderInline(m.text || '');
      row.appendChild(bubble);
      wrap.appendChild(row);
    });
    if (!(d.messages || []).length) {
      stateEl.textContent = 'This chat has no messages yet.';
      stateEl.style.display = 'block';
    } else {
      document.getElementById('continue-bar').style.display = 'block';
    }
  } catch (e) {
    stateEl.textContent = 'Network error loading this shared chat.';
    stateEl.style.display = 'block';
  }
})();

const continueBtn = document.getElementById('continue-btn');
const continueErr = document.getElementById('continue-error');
continueBtn.addEventListener('click', async () => {
  continueErr.style.display = 'none';
  continueBtn.disabled = true;
  const origHTML = continueBtn.innerHTML;
  continueBtn.textContent = 'Setting up your copy…';
  try {
    const r = await fetch('/api/share/' + encodeURIComponent(shareId) + '/continue', { method: 'POST' });
    const d = await r.json();
    if (!r.ok || d.error || !d.conversation_id) {
      continueErr.textContent = d.error || 'Could not continue this chat right now.';
      continueErr.style.display = 'block';
      continueBtn.disabled = false;
      continueBtn.innerHTML = origHTML;
      return;
    }
    // This becomes the visitor's OWN private, editable copy — the
    // original owner's conversation is never touched.
    location.href = '/?open=' + encodeURIComponent(d.conversation_id);
  } catch (e) {
    continueErr.textContent = 'Network error: ' + e.message;
    continueErr.style.display = 'block';
    continueBtn.disabled = false;
    continueBtn.innerHTML = origHTML;
  }
});
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
      badge:   '/badge.png',
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

@app.route("/badge.png")
def pwa_badge():
    """Badge icon for push notifications (status bar) — same as regular icon."""
    return Response(_get_icon(96), mimetype="image/png",
                    headers={"Cache-Control": "public, max-age=604800"})

@app.route("/favicon.ico")
def favicon():
    return Response(_get_icon(192), mimetype="image/png",
                    headers={"Cache-Control": "public, max-age=604800"})


@app.route("/")
@login_required
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")


@app.route("/api/invite-link", methods=["GET"])
@login_required
def api_invite_link():
    code = get_or_create_invite_code()
    # Adopt the caller's own existing user_id as the "owner" account the
    # first time this is ever called, so it's YOUR real chat history that
    # gets shared via the link — not a fresh empty account.
    get_or_create_owner_id(preferred_id=current_username())
    return jsonify({"invite_url": request.host_url.rstrip("/") + "/invite/" + code})


@app.route("/invite/<code>")
def invite_landing(code):
    # Anyone opening this link gets logged into the OWNER's account, so they
    # see and can add to the same chat history — this is an intentional
    # shared-account link, not a per-visitor anonymous session like the bare
    # domain gives. The code itself isn't checked against anything (there's
    # no per-invite access control here) — treat this link as equivalent to
    # sharing your password, and only send it to people you trust with full
    # access to your chats.
    session["user_id"] = get_or_create_owner_id()
    session.permanent = True
    return Response(PAGE, mimetype="text/html; charset=utf-8")


# --- Claim owner status (needed on serverless hosts like Vercel) -------------
# On Vercel, each request can land on a different instance with its own
# ephemeral /tmp, so a locally-written owner_user_id.txt does NOT reliably
# make you "the owner" on every subsequent request — see get_or_create_owner_id
# above. If OWNER_USER_ID is set (a fixed id you choose, e.g. a UUID), that
# value is always returned as the owner id, no file/DB needed. This route
# lets YOUR browser's session get pinned to that exact id, once, so every
# future request from you compares equal to it. Protect it with OWNER_SECRET
# (also an env var) so nobody else can claim it.
#
# Setup on Vercel:
#   1. Generate two random values locally:
#        python -c "import secrets,uuid; print('OWNER_USER_ID=', uuid.uuid4()); print('OWNER_SECRET=', secrets.token_hex(16))"
#   2. Set OWNER_USER_ID, OWNER_SECRET, and FLASK_SECRET_KEY as environment
#      variables in your Vercel project settings, then redeploy.
#   3. Visit https://<your-app>.vercel.app/claim-owner/<OWNER_SECRET> once,
#      in the browser you want to use as the owner. You only need to do
#      this again if you ever clear cookies or switch browsers/devices.
@app.route("/claim-owner/<secret>")
def claim_owner(secret):
    owner_secret = _os.environ.get("OWNER_SECRET", "").strip()
    if not owner_secret:
        return Response(
            "OWNER_SECRET is not set on the server, so no one can claim "
            "ownership this way. Set OWNER_USER_ID, OWNER_SECRET, and "
            "FLASK_SECRET_KEY as environment variables first, then reload "
            "this page.", mimetype="text/plain"), 400
    if not secrets.compare_digest(secret, owner_secret):
        return Response("Incorrect secret.", mimetype="text/plain"), 403
    if not _OWNER_USER_ID_ENV:
        return Response(
            "OWNER_SECRET is set but OWNER_USER_ID is not. Set OWNER_USER_ID "
            "too (any fixed string, e.g. a UUID), then reload this page.",
            mimetype="text/plain"), 400
    session["user_id"] = _OWNER_USER_ID_ENV
    session.permanent = True
    return Response(
        "You're now recognized as the account owner on this browser. "
        "<a href='/api-usage'>Go to API keys →</a>",
        mimetype="text/html; charset=utf-8")


# --- API key management (open to everyone — no owner gate) --------------------
@app.route("/api/keys", methods=["GET"])
@login_required
def api_keys_list():
    return jsonify({"keys": list_api_keys(current_username())})

def _fmt_dt(iso_str):
    if not iso_str:
        return "Never"
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %-d, %Y, %-I:%M %p")
    except Exception:
        return iso_str

@app.route("/api-usage")
@login_required
def api_usage_page():
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>API Keys · Mythic AI</title>
<style>
  * { box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
          background:#0f1115; color:#f2f2f2; margin:0; padding:32px 24px 60px; }
  .wrap { max-width:1000px; margin:0 auto; }
  a.back { color:#9a9ea6; text-decoration:none; font-size:14px; display:inline-block; margin-bottom:20px; }
  a.back:hover { color:#fff; }
  .headrow { display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:14px; margin-bottom:8px; }
  h1 { font-size:30px; margin:0 0 6px; }
  .sub { color:#9a9ea6; font-size:14px; margin-bottom:30px; max-width:560px; line-height:1.5; }
  .gen-btn { background:#e8532a; color:#fff; border:none; border-radius:8px; padding:11px 18px;
             font-size:13px; font-weight:700; letter-spacing:.3px; cursor:pointer; white-space:nowrap; }
  .gen-btn:hover { background:#d1471f; }
  .totals { display:flex; gap:16px; margin-bottom:30px; flex-wrap:wrap; }
  .totals .box { flex:1; min-width:150px; background:#1a1d24; border:1px solid #2a2e37;
                   border-radius:14px; padding:18px; text-align:center; }
  .totals .num { font-size:40px; font-weight:800; line-height:1.1; }
  .totals .label { font-size:12px; color:#9a9ea6; margin-top:6px; letter-spacing:.3px; text-transform:uppercase; }
  table { width:100%; border-collapse:collapse; background:#1a1d24; border:1px solid #2a2e37; border-radius:14px; overflow:hidden; }
  thead th { text-align:left; font-size:11px; letter-spacing:.5px; text-transform:uppercase; color:#9a9ea6;
             padding:14px 16px; border-bottom:1px solid #2a2e37; }
  tbody td { padding:16px; border-bottom:1px solid #22252c; font-size:14px; vertical-align:middle; }
  tbody tr:last-child td { border-bottom:none; }
  tbody tr:hover { background:#20232b; }
  .name-cell { font-weight:700; font-size:15px; }
  .key-cell { font-family:monospace; color:#c7cad1; display:flex; align-items:center; gap:8px; }
  .copy-btn { background:none; border:1px solid #3a3e47; color:#c7cad1; cursor:pointer; font-size:11.5px;
              padding:4px 9px; border-radius:6px; font-weight:600; white-space:nowrap; }
  .copy-btn:hover { border-color:#7be3ab; color:#7be3ab; }
  .calls-cell { font-weight:700; font-size:18px; }
  .state-pill { font-size:11px; font-weight:700; letter-spacing:.4px; border:1px solid; border-radius:20px; padding:3px 10px; display:inline-block; }
  .state-active { color:#1a9e5c; border-color:#1a9e5c; }
  .state-revoked { color:#c0392b; border-color:#c0392b; }
  .revoke-btn { background:none; border:1px solid #3a3e47; color:#c0392b; border-radius:6px; padding:6px 10px;
                font-size:12px; cursor:pointer; }
  .revoke-btn:hover { background:#c0392b; color:#fff; border-color:#c0392b; }
  .rename-btn { background:none; border:1px solid #3a3e47; color:#9a9ea6; border-radius:6px; padding:6px 10px;
                font-size:12px; cursor:pointer; margin-right:6px; }
  .rename-btn:hover { border-color:#e8532a; color:#e8532a; }
  .options-cell { display:flex; gap:6px; flex-wrap:wrap; }
  .empty { color:#9a9ea6; font-size:15px; padding:50px 0; text-align:center; }
  .modal-overlay { display:none; position:fixed; inset:0; background:#000a; align-items:center; justify-content:center; z-index:50; }
  .modal { background:#1a1d24; border:1px solid #2a2e37; border-radius:14px; padding:26px; width:min(90vw,420px); }
  .modal h3 { margin:0 0 14px; font-size:18px; }
  .modal input { width:100%; padding:10px 12px; border-radius:8px; border:1px solid #3a3e47; background:#0f1115;
                 color:#fff; font-size:14px; margin-bottom:14px; }
  .modal-actions { display:flex; gap:10px; justify-content:flex-end; }
  .modal-actions button { border-radius:8px; padding:9px 16px; font-size:13px; cursor:pointer; border:none; }
  .btn-cancel { background:#2a2e37; color:#f2f2f2; }
  .btn-confirm { background:#e8532a; color:#fff; font-weight:700; }
  .new-key-box { background:#0f1115; border:1px solid #1a9e5c; border-radius:8px; padding:12px; margin-bottom:14px;
                 font-family:monospace; font-size:13px; word-break:break-all; color:#7be3ab; }
</style></head>
<body>
  <div class="wrap">
    <a class="back" href="/">← Back to chat</a>
    <div class="headrow">
      <div>
        <h1>API keys</h1>
        <div class="sub">Create and manage API keys for authenticating requests to Mythic AI. These keys allow programmatic access to your app.</div>
      </div>
      <button class="gen-btn" onclick="openCreateModal()">GENERATE API KEY</button>
    </div>
    <div class="totals" id="totals-row"></div>
    <table>
      <thead><tr>
        <th>Name</th><th>API Key</th><th>Created At</th><th>Calls</th><th>State</th><th>Options</th>
      </tr></thead>
      <tbody id="keys-tbody"></tbody>
    </table>
    <div class="empty" id="empty-msg" style="display:none;">No API keys yet. Click "Generate API Key" to create one.</div>
  </div>

  <div class="modal-overlay" id="create-overlay">
    <div class="modal">
      <h3>Generate API key</h3>
      <input type="text" id="create-label" placeholder="Key name (optional)" maxlength="100">
      <div id="new-key-result"></div>
      <div class="modal-actions">
        <button class="btn-cancel" onclick="closeCreateModal()">Close</button>
        <button class="btn-confirm" id="create-confirm-btn" onclick="doCreateKey()">Generate</button>
      </div>
    </div>
  </div>

<script>
function fmtDate(iso) {
  if (!iso) return 'Never';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString(undefined, { month:'short', day:'numeric', year:'numeric', hour:'numeric', minute:'2-digit' });
}

async function loadKeys() {
  const res = await fetch('/api/keys');
  if (res.status === 403) {
    document.querySelector('.wrap').innerHTML = '<a class="back" href="/">← Back to chat</a><div class="empty">Only the account owner can view API keys.</div>';
    return;
  }
  const data = await res.json();
  const keys = data.keys || [];
  const totalsRow = document.getElementById('totals-row');
  const tbody = document.getElementById('keys-tbody');
  const emptyMsg = document.getElementById('empty-msg');

  const activeCount = keys.filter(k => k.active).length;
  const totalCalls = keys.reduce((s,k) => s + (k.request_count || 0), 0);
  totalsRow.innerHTML = `
    <div class="box"><div class="num">${activeCount}</div><div class="label">Active Keys</div></div>
    <div class="box"><div class="num">${totalCalls}</div><div class="label">Total Calls</div></div>
    <div class="box"><div class="num">${keys.length}</div><div class="label">Total Keys</div></div>`;

  if (!keys.length) {
    tbody.innerHTML = '';
    emptyMsg.style.display = 'block';
    return;
  }
  emptyMsg.style.display = 'none';

  tbody.innerHTML = keys.map(k => `
    <tr>
      <td class="name-cell" id="name-cell-${k.id}">${(k.label || '(unnamed key)').replace(/</g,'&lt;')}</td>
      <td><div class="key-cell"><span>${k.key_prefix || ''}</span>
        <button class="copy-btn" title="Copy key prefix" onclick="copyKeyPrefix('${(k.key_prefix||'').replace(/'/g,"")}', this)">📋 Copy</button></div></td>
      <td>${fmtDate(k.created_at)}</td>
      <td class="calls-cell">${k.request_count || 0}</td>
      <td><span class="state-pill ${k.active ? 'state-active' : 'state-revoked'}">${k.active ? 'ACTIVE' : 'REVOKED'}</span></td>
      <td><div class="options-cell">
        <button class="rename-btn" onclick="renameKey('${k.id}', ${JSON.stringify(k.label || '')})">✎ Rename</button>
        ${k.active ? `<button class="revoke-btn" onclick="revokeKey('${k.id}')">Revoke</button>` : ''}
      </div></td>
    </tr>`).join('');
}

function copyKeyPrefix(prefix, btn) {
  navigator.clipboard.writeText(prefix).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✓ Copied';
    setTimeout(() => { btn.textContent = orig; }, 1200);
  });
}

async function renameKey(id, currentLabel) {
  const newLabel = prompt('New name for this key:', currentLabel || '');
  if (newLabel === null) return;
  const trimmed = newLabel.trim();
  if (!trimmed) { alert('Name cannot be empty.'); return; }
  const res = await fetch('/api/keys/' + id, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label: trimmed }),
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  loadKeys();
}

async function revokeKey(id) {
  if (!confirm('Revoke this key? Apps using it will stop working immediately.')) return;
  await fetch('/api/keys/' + id, { method: 'DELETE' });
  loadKeys();
}

function openCreateModal() {
  document.getElementById('create-label').value = '';
  document.getElementById('new-key-result').innerHTML = '';
  document.getElementById('create-overlay').style.display = 'flex';
}
function closeCreateModal() {
  document.getElementById('create-overlay').style.display = 'none';
  loadKeys();
}
async function doCreateKey() {
  const label = document.getElementById('create-label').value.trim();
  const btn = document.getElementById('create-confirm-btn');
  btn.disabled = true;
  try {
    const res = await fetch('/api/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    const data = await res.json();
    if (data.api_key) {
      document.getElementById('new-key-result').innerHTML =
        '<div class="new-key-box" style="display:flex;gap:8px;align-items:center;">' +
          '<input type="text" id="standalone-new-key-input" readonly value="' + data.api_key.replace(/"/g, '&quot;') + '" ' +
            'style="flex:1;background:transparent;border:none;color:#7be3ab;font-family:monospace;font-size:13px;min-width:0;">' +
          '<button type="button" id="standalone-new-key-copy-btn" ' +
            'style="background:#1a9e5c;color:#04140b;border:none;padding:8px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:700;white-space:nowrap;">Copy</button>' +
        '</div>' +
        '<div style="font-size:12px;color:#9a9ea6;margin-bottom:10px;">Copy this now — it will not be shown again. (Use the Copy button — double-clicking to select may only grab part of the key due to the hyphens.)</div>';
      const stCopyBtn = document.getElementById('standalone-new-key-copy-btn');
      if (stCopyBtn) {
        stCopyBtn.addEventListener('click', async () => {
          const keyInput = document.getElementById('standalone-new-key-input');
          if (!keyInput) return;
          try {
            keyInput.select();
            document.execCommand('copy');
            stCopyBtn.textContent = '✓ Copied!';
            setTimeout(() => { stCopyBtn.textContent = 'Copy'; }, 2500);
          } catch (e) {
            try {
              await navigator.clipboard.writeText(keyInput.value);
              stCopyBtn.textContent = '✓ Copied!';
              setTimeout(() => { stCopyBtn.textContent = 'Copy'; }, 2500);
            } catch (e2) {
              alert('Copy failed. Click once inside the box, then Ctrl+A, Ctrl+C.');
            }
          }
        });
      }
      loadKeys();
    } else if (data.error) {
      alert(data.error);
    }
  } catch (e) {
    alert('Could not create key.');
  } finally {
    btn.disabled = false;
  }
}

loadKeys();
</script>
</body></html>"""
    return Response(html, mimetype="text/html; charset=utf-8")


@app.route("/analytics")
@login_required
def analytics_dashboard():
    """Comprehensive analytics dashboard for viewing usage stats, trends, and exporting conversations."""
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analytics Dashboard · Mythic AI</title>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f1115;
      color: #f2f2f2;
      padding: 32px 20px;
    }
    .wrap { max-width: 1200px; margin: 0 auto; }
    h1 { font-size: 28px; margin: 0 0 8px; font-weight: 700; }
    .subtitle { color: #9a9ea6; font-size: 14px; margin-bottom: 30px; }
    
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }
    .metric {
      background: #1a1d24;
      border: 1px solid #2a2e37;
      border-radius: 12px;
      padding: 20px;
    }
    .metric-value { font-size: 32px; font-weight: 800; margin-bottom: 6px; }
    .metric-label { font-size: 11px; color: #9a9ea6; text-transform: uppercase; }
    
    .tabs {
      display: flex;
      gap: 0;
      border-bottom: 1px solid #2a2e37;
      margin-bottom: 20px;
    }
    .tab-button {
      padding: 12px 18px;
      background: none;
      border: none;
      color: #9a9ea6;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }
    .tab-button:hover { color: #fff; }
    .tab-button.active { color: #fff; border-bottom-color: #e8532a; }
    
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    .card {
      background: #1a1d24;
      border: 1px solid #2a2e37;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 16px;
    }
    
    .search-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }
    .search-row input,
    .search-row select {
      padding: 10px 14px;
      border: 1px solid #3a3e47;
      background: #0f1115;
      color: #fff;
      border-radius: 8px;
      font-size: 13px;
      font-family: inherit;
    }
    .search-row button {
      padding: 10px 24px;
      background: #e8532a;
      color: #fff;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 700;
      font-size: 13px;
      transition: all 0.2s;
    }
    .search-row button:hover { background: #d1471f; }
    
    .results {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 500px;
      overflow-y: auto;
    }
    .result {
      background: #0f1115;
      border: 1px solid #2a2e37;
      border-radius: 8px;
      padding: 12px 14px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .result:hover { border-color: #e8532a; background: #1a1d24; }
    .result-title { font-weight: 700; margin-bottom: 4px; }
    .result-meta { font-size: 12px; color: #9a9ea6; }
    
    .export-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 12px;
    }
    .export-item {
      background: #1a1d24;
      border: 1px solid #2a2e37;
      border-radius: 8px;
      padding: 12px;
      text-align: center;
      cursor: pointer;
      font-weight: 700;
      font-size: 13px;
      transition: all 0.2s;
    }
    .export-item:hover { border-color: #e8532a; color: #e8532a; }
    
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    thead { background: #0f1115; }
    th {
      text-align: left;
      padding: 12px;
      font-size: 11px;
      color: #9a9ea6;
      text-transform: uppercase;
      font-weight: 600;
      border-bottom: 1px solid #2a2e37;
    }
    td { padding: 12px; border-bottom: 1px solid #2a2e37; }
    tr:hover { background: #1a1d24; }
    
    .loading { text-align: center; color: #9a9ea6; padding: 40px 20px; }
    .error { color: #ef4444; padding: 12px; background: #2a1515; border-radius: 8px; }
    
    .format-select {
      padding: 10px 14px;
      border: 1px solid #3a3e47;
      background: #0f1115;
      color: #fff;
      border-radius: 8px;
      font-size: 13px;
      font-family: inherit;
      width: 200px;
      margin-bottom: 20px;
    }
  </style>
</head>
<body>

<div class="wrap">
  <h1>📊 Analytics Dashboard</h1>
  <p class="subtitle">View usage statistics, search conversations, and export your data</p>

  <div class="metrics">
    <div class="metric">
      <div class="metric-value" id="metric-chats">-</div>
      <div class="metric-label">Total Chats</div>
    </div>
    <div class="metric">
      <div class="metric-value" id="metric-messages">-</div>
      <div class="metric-label">Total Messages</div>
    </div>
    <div class="metric">
      <div class="metric-value" id="metric-folders">-</div>
      <div class="metric-label">Folders</div>
    </div>
    <div class="metric">
      <div class="metric-value" id="metric-week">-</div>
      <div class="metric-label">This Week</div>
    </div>
  </div>

  <div class="tabs">
    <button class="tab-button active" onclick="showTab('search')">🔎 Search</button>
    <button class="tab-button" onclick="showTab('export')">📥 Export</button>
    <button class="tab-button" onclick="showTab('trends')">📈 Trends</button>
  </div>

  <!-- SEARCH TAB -->
  <div id="search" class="tab-content active">
    <div class="card">
      <h2 style="margin-top: 0;">Search Conversations</h2>
      <div class="search-row">
        <input type="text" id="search-query" placeholder="Search..." />
        <input type="date" id="search-start" />
        <input type="date" id="search-end" />
        <select id="search-folder">
          <option value="">All Folders</option>
        </select>
        <button onclick="performSearch()">Search</button>
      </div>
      <div id="search-results" class="results"></div>
    </div>
  </div>

  <!-- EXPORT TAB -->
  <div id="export" class="tab-content">
    <div class="card">
      <h2 style="margin-top: 0;">Export Conversations</h2>
      <label style="display: block; margin-bottom: 8px; font-weight: 700;">Format:</label>
      <select id="export-format" class="format-select">
        <option value="json">JSON</option>
        <option value="html">HTML</option>
        <option value="csv">CSV</option>
      </select>
      <div id="export-list" class="export-grid"></div>
    </div>
  </div>

  <!-- TRENDS TAB -->
  <div id="trends" class="tab-content">
    <div class="card">
      <h2 style="margin-top: 0;">Usage Trends (Last 30 Days)</h2>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Requests</th>
            <th>Tokens</th>
            <th>Users</th>
          </tr>
        </thead>
        <tbody id="trends-table"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
  // Show/hide tabs
  function showTab(tabName) {
    // Hide all tabs
    var tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(function(tab) {
      tab.classList.remove('active');
    });
    
    // Deactivate all buttons
    var btns = document.querySelectorAll('.tab-button');
    btns.forEach(function(btn) {
      btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Activate clicked button
    event.target.classList.add('active');
    
    // Load data if needed
    if (tabName === 'trends') {
      loadTrends();
    }
    if (tabName === 'export') {
      loadExportList();
    }
  }

  // Load initial dashboard
  function loadDashboard() {
    fetch('/api/analytics/dashboard')
      .then(r => r.json())
      .then(data => {
        document.getElementById('metric-chats').textContent = data.dashboard.total_conversations;
        document.getElementById('metric-messages').textContent = data.dashboard.total_messages_sent;
        document.getElementById('metric-folders').textContent = Object.keys(data.dashboard.folders_breakdown).length;
        document.getElementById('metric-week').textContent = data.dashboard.usage_this_week.total_requests;
        
        // Fill folder dropdown
        var folderSelect = document.getElementById('search-folder');
        Object.keys(data.dashboard.folders_breakdown).forEach(function(folder) {
          var opt = document.createElement('option');
          opt.value = folder;
          opt.textContent = folder + ' (' + data.dashboard.folders_breakdown[folder] + ')';
          folderSelect.appendChild(opt);
        });
      })
      .catch(function(err) {
        console.error('Failed to load dashboard:', err);
      });
  }

  // Search conversations
  function performSearch() {
    var query = document.getElementById('search-query').value;
    var startDate = document.getElementById('search-start').value;
    var endDate = document.getElementById('search-end').value;
    var folder = document.getElementById('search-folder').value;
    
    var resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = '<div class="loading">Searching...</div>';
    
    var filters = {};
    if (startDate) filters.start_date = startDate;
    if (endDate) filters.end_date = endDate;
    if (folder) filters.folder = folder;
    
    fetch('/api/analytics/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, filters: filters })
    })
      .then(r => r.json())
      .then(data => {
        if (data.found === 0) {
          resultsDiv.innerHTML = '<div class="loading">No results found</div>';
          return;
        }
        
        var html = '';
        data.conversations.forEach(function(conv) {
          html += '<div class="result" onclick="exportConv(\'' + conv.id + '\')">';
          html += '<div class="result-title">' + (conv.title || 'Untitled').substring(0, 50) + '</div>';
          html += '<div class="result-meta">' + (conv.messages ? conv.messages.length : 0) + ' messages</div>';
          html += '</div>';
        });
        resultsDiv.innerHTML = html;
      })
      .catch(function(err) {
        resultsDiv.innerHTML = '<div class="error">Error: ' + err.message + '</div>';
      });
  }

  // Load export list
  function loadExportList() {
    var listDiv = document.getElementById('export-list');
    listDiv.innerHTML = '<div class="loading">Loading conversations...</div>';
    
    fetch('/api/analytics/dashboard')
      .then(r => r.json())
      .then(data => {
        var convs = data.dashboard.recent_conversations || [];
        if (convs.length === 0) {
          listDiv.innerHTML = '<div class="loading">No conversations</div>';
          return;
        }
        
        var html = '';
        convs.forEach(function(conv) {
          html += '<div class="export-item" onclick="exportConv(\'' + conv.id + '\')">';
          html += (conv.title || 'Untitled').substring(0, 20);
          html += '</div>';
        });
        listDiv.innerHTML = html;
      })
      .catch(function(err) {
        listDiv.innerHTML = '<div class="error">Error: ' + err.message + '</div>';
      });
  }

  // Export a conversation
  function exportConv(convId) {
    var format = document.getElementById('export-format').value;
    var url = '/api/conversations/' + encodeURIComponent(convId) + '/export?format=' + format;
    window.location.href = url;
  }

  // Load trends
  function loadTrends() {
    var tbody = document.getElementById('trends-table');
    tbody.innerHTML = '<tr><td colspan="4" class="loading">Loading trends...</td></tr>';
    
    fetch('/api/analytics/trend?days=30')
      .then(r => r.json())
      .then(data => {
        if (!data.trend || data.trend.length === 0) {
          tbody.innerHTML = '<tr><td colspan="4" class="loading">No trend data</td></tr>';
          return;
        }
        
        var html = '';
        data.trend.forEach(function(row) {
          html += '<tr>';
          html += '<td>' + row.date + '</td>';
          html += '<td>' + row.requests + '</td>';
          html += '<td>' + row.tokens + '</td>';
          html += '<td>' + row.unique_users + '</td>';
          html += '</tr>';
        });
        tbody.innerHTML = html;
      })
      .catch(function(err) {
        tbody.innerHTML = '<tr><td colspan="4" class="error">Error: ' + err.message + '</td></tr>';
      });
  }

  // Allow Enter in search
  document.addEventListener('DOMContentLoaded', function() {
    var searchInput = document.getElementById('search-query');
    if (searchInput) {
      searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
          performSearch();
        }
      });
    }
  });

  // Load on startup
  loadDashboard();
</script>

</body>
</html>"""
    return Response(html, mimetype="text/html; charset=utf-8")






@login_required
def api_keys_create():
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()
    raw_key, record = create_api_key(label, current_username())
    return jsonify({
        "api_key": raw_key,  # shown ONCE — the frontend must display and never re-fetch this
        "id": record["id"],
        "key_prefix": record["key_prefix"],
        "label": record["label"],
        "created_at": record["created_at"],
    })

@app.route("/api/keys/<key_id>", methods=["DELETE"])
@login_required
def api_keys_revoke(key_id):
    ok = revoke_api_key(key_id, current_username())
    return jsonify({"revoked": ok})

@app.route("/api/keys/<key_id>", methods=["PATCH"])
@login_required
def api_keys_rename(key_id):
    data = request.get_json(silent=True) or {}
    new_label = (data.get("label") or "").strip()
    if not new_label:
        return jsonify({"error": "label is required"}), 400
    ok = rename_api_key(key_id, new_label, current_username())
    if not ok:
        return jsonify({"error": "key not found"}), 404
    return jsonify({"renamed": ok, "label": new_label})


# ══════════════════════════════════════════════════════════════════════════════
# ─── ANALYTICS & USAGE TRACKING API ENDPOINTS ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/analytics/usage", methods=["GET"])
@login_required
def api_get_usage_report():
    """Get detailed usage analytics for authenticated user's API keys and activity.
    Query params: days_back (default 30), start_date, end_date, key_id (filter by specific key)"""
    username = current_username()
    days_back = request.args.get("days_back", 30, type=int)
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    key_id = request.args.get("key_id")
    
    report = get_usage_report(username=username, key_id=key_id, start_date=start_date, 
                              end_date=end_date, days_back=days_back)
    return jsonify(report)


@app.route("/api/analytics/search", methods=["POST"])
@login_required
def api_search_conversations():
    """Full-text search conversations with advanced filtering.
    POST body: { 'query': 'string', 'filters': { 'start_date': 'YYYY-MM-DD', 
    'end_date': 'YYYY-MM-DD', 'folder': 'name', 'archived': bool, 
    'pinned': bool, 'min_messages': int } }"""
    username = current_username()
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    filters = data.get("filters", {})
    
    results = search_conversations(username, query=query, filters=filters)
    return jsonify({"found": len(results), "conversations": results})


@app.route("/api/conversations/<conv_id>/export", methods=["GET"])
@login_required
def api_export_conversation(conv_id):
    """Export a conversation in requested format (json, csv, html).
    Query param: format (default 'json')"""
    username = current_username()
    format_type = request.args.get("format", "json").lower()
    
    if format_type not in ("json", "csv", "html"):
        return jsonify({"error": "format must be json, csv, or html"}), 400
    
    content, mimetype, filename = export_conversation(conv_id, username, format_type)
    if not content:
        return jsonify({"error": "conversation not found"}), 404
    
    return Response(content, mimetype=mimetype, 
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.route("/api/analytics/dashboard", methods=["GET"])
@login_required
def api_analytics_dashboard():
    """Get a quick dashboard summary of user's activity.
    Includes: recent activity, top models, conversation trends."""
    username = current_username()
    report = get_usage_report(username=username, days_back=7)
    
    # Get recent conversations
    convs = list_conversations(username)
    recent = sorted(convs, key=lambda c: c.get("updated_at", ""), reverse=True)[:10]
    
    # Get folder breakdown
    folders = {}
    for conv in convs:
        folder = conv.get("folder") or "Uncategorized"
        if folder not in folders:
            folders[folder] = 0
        folders[folder] += 1
    
    return jsonify({
        "username": username,
        "dashboard": {
            "total_conversations": len(convs),
            "total_messages_sent": sum(len(c.get("messages", [])) for c in convs),
            "recent_conversations": recent,
            "folders_breakdown": folders,
            "usage_this_week": report["summary"],
            "generated_at": _dt.utcnow().isoformat()
        }
    })


@app.route("/api/admin/stats", methods=["GET"])
@login_required
def api_admin_stats():
    """Get system-wide admin statistics (requires admin role).
    Includes: total users, total requests, top models, daily trends."""
    username = current_username()
    
    # Check if user is admin (you can add a proper admin role check here)
    # For now, allow super-admin access
    is_admin = username in getattr(_app, 'SUPER_ADMINS', [])
    if not is_admin and os.environ.get("SUPER_ADMIN_USERNAME") != username:
        return jsonify({"error": "admin access required"}), 403
    
    stats = get_admin_stats(include_daily=True)
    return jsonify(stats)


@app.route("/api/analytics/export-bulk", methods=["POST"])
@login_required
def api_export_bulk():
    """Export multiple conversations at once in ZIP format.
    POST body: { 'conversation_ids': ['id1', 'id2', ...], 'format': 'json|csv|html' }"""
    username = current_username()
    data = request.get_json(silent=True) or {}
    conv_ids = data.get("conversation_ids", [])
    format_type = data.get("format", "json").lower()
    
    if not conv_ids:
        return jsonify({"error": "conversation_ids required"}), 400
    if format_type not in ("json", "csv", "html"):
        return jsonify({"error": "format must be json, csv, or html"}), 400
    
    # Create ZIP in memory
    import zipfile as _zipfile
    from io import BytesIO as _BytesIO
    
    zip_buffer = _BytesIO()
    with _zipfile.ZipFile(zip_buffer, "w", _zipfile.ZIP_DEFLATED) as zip_file:
        for conv_id in conv_ids:
            content, mimetype, filename = export_conversation(conv_id, username, format_type)
            if content:
                zip_file.writestr(filename, content)
    
    zip_buffer.seek(0)
    return Response(zip_buffer.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="conversations.zip"'})


@app.route("/api/analytics/filters-options", methods=["GET"])
@login_required
def api_search_filter_options():
    """Get available filter options for conversation search (folders, date range, etc)."""
    username = current_username()
    convs = list_conversations(username)
    
    folders = set(c.get("folder") for c in convs if c.get("folder"))
    folders.add("Uncategorized")
    
    dates = [c.get("updated_at", "")[:10] for c in convs if c.get("updated_at")]
    min_date = min(dates) if dates else None
    max_date = max(dates) if dates else None
    
    return jsonify({
        "available_folders": list(folders),
        "date_range": {"min": min_date, "max": max_date},
        "total_conversations": len(convs),
        "filters_available": ["start_date", "end_date", "folder", "archived", "pinned", "min_messages"]
    })


@app.route("/api/analytics/trend", methods=["GET"])
@login_required
def api_usage_trend():
    """Get usage trends over time (daily breakdown).
    Query params: days (default 30)"""
    username = current_username()
    days = request.args.get("days", 30, type=int)
    
    report = get_usage_report(username=username, days_back=days)
    trend_data = []
    
    for day_entry in report["summary"]["daily_breakdown"]:
        date = day_entry["date"]
        stats = day_entry["stats"]
        trend_data.append({
            "date": date,
            "requests": stats.get("total_requests", 0),
            "tokens": stats.get("total_tokens", 0),
            "unique_users": len(stats.get("unique_users", []))
        })
    
    return jsonify({
        "period_days": days,
        "trend": trend_data,
        "generated_at": _dt.utcnow().isoformat()
    })


@app.route("/v1/chat/completions", methods=["POST"])
def v1_chat_completions():
    auth = request.headers.get("Authorization", "")
    raw_key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not verify_api_key(raw_key):
        return jsonify({"error": {"message": "Invalid or missing API key. Include "
                                              "'Authorization: Bearer aarav-...'.",
                                   "type": "invalid_request_error"}}), 401

    data = request.get_json(silent=True) or {}
    incoming = data.get("messages") or []
    if not incoming:
        return jsonify({"error": {"message": "'messages' is required.",
                                   "type": "invalid_request_error"}}), 400

    system_prompt = SYSTEM_PROMPT
    gemini_messages = []
    for m in incoming:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_prompt = content or system_prompt
            continue
        gemini_messages.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": content}],
        })

    stream_requested = bool(data.get("stream"))
    model_name = data.get("model") or "mythic-2"

    if stream_requested:
        def sse():
            for chunk in auto_stream_chunks(None, gemini_messages, system_prompt):
                payload = {
                    "id": "chatcmpl-" + uuid.uuid4().hex[:24],
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(sse()), mimetype="text/event-stream")

    full_reply = "".join(auto_stream_chunks(None, gemini_messages, system_prompt))
    return jsonify({
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_reply},
            "finish_reason": "stop",
        }],
        "usage": {  # not tracked precisely — placeholder for client compatibility
            "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
        },
    })


# --- Public API: /v1/images/generations (OpenAI-compatible, requires API key) -----
@app.route("/v1/images/generations", methods=["POST"])
def v1_images_generations():
    """OpenAI-compatible endpoint: POST with prompt + style, get back image URLs."""
    auth = request.headers.get("Authorization", "")
    raw_key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not verify_api_key(raw_key):
        return jsonify({"error": {"message": "Invalid or missing API key.",
                                   "type": "invalid_request_error"}}), 401
    
    # Reuse the existing /api/generate-image logic by calling it internally
    # We'll just set the session and call the endpoint function
    session["user_id"] = get_or_create_owner_id(preferred_id=raw_key)
    return generate_image()


# --- Public API: /v1/code/execute (OpenAI-compatible, requires API key) ----------
@app.route("/v1/code/execute", methods=["POST"])
def v1_code_execute():
    """OpenAI-compatible endpoint: POST with code + language, get back output/errors."""
    auth = request.headers.get("Authorization", "")
    raw_key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not verify_api_key(raw_key):
        return jsonify({"error": {"message": "Invalid or missing API key.",
                                   "type": "invalid_request_error"}}), 401
    
    # Reuse the existing /api/execute-code logic
    session["user_id"] = get_or_create_owner_id(preferred_id=raw_key)
    return api_execute_code()


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


@app.route("/api/conversations", methods=["POST"])
@login_required
def api_create_conversation():
    """Creates an empty 'New chat' conversation immediately, so it shows up
    in the sidebar right away instead of only appearing after the first
    message is sent. The conversation carries no messages yet — sending the
    first message into it works exactly the same as any other conversation."""
    username = current_username()
    conv_id = str(uuid.uuid4())
    save_conversation(username, conv_id, {"title": "New chat", "messages": []})
    return jsonify({"id": conv_id, "title": "New chat"})


# Patterns that identify a conversation as internal-tooling leakage rather
# than a real chat — e.g. old follow-up-suggestion or tone/length-prefix
# requests saved before the ephemeral-request fix existed. Matched against
# either the saved title or the first user message.
_JUNK_CONV_PATTERNS = (
    "based on this ai reply, suggest",
    "[instructions:",
)


def _conv_is_junk(conv_summary, username):
    title = (conv_summary.get("title") or "").strip().lower()
    if any(title.startswith(p) for p in _JUNK_CONV_PATTERNS):
        return True
    # Title alone might be a truncated/renamed version — check the first
    # real user message too, for conversations saved before titles were
    # cleaned up server-side.
    full = load_conversation(username, conv_summary["id"])
    if not full:
        return False
    for m in full.get("messages", []):
        if m.get("role") != "user":
            continue
        text = "".join(p.get("text", "") for p in m.get("parts", []) if "text" in p).strip().lower()
        if text:
            return any(text.startswith(p) for p in _JUNK_CONV_PATTERNS)
    return False


@app.route("/api/conversations/cleanup-junk", methods=["POST"])
@login_required
def api_cleanup_junk_conversations():
    """Finds and deletes stray conversations created by internal tooling
    (old follow-up-suggestion / tone-prefix requests) rather than real user
    chats — see _JUNK_CONV_PATTERNS. Safe to call any time; only ever
    deletes conversations matching those specific patterns."""
    username = current_username()
    convs = list_conversations(username)
    removed = []
    for c in convs:
        if _conv_is_junk(c, username):
            delete_conversation(username, c["id"])
            removed.append(c.get("title", c["id"]))
    return jsonify({"status": "done", "removed_count": len(removed), "removed_titles": removed})


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


def _share_url_for(share_id):
    return request.host_url.rstrip("/") + "/share/" + share_id


@app.route("/api/conversations/<conv_id>/share", methods=["GET"])
@login_required
def api_get_share_status(conv_id):
    """Returns whether this conversation currently has an active public
    share link, and its URL if so."""
    username = current_username()
    if load_conversation(username, conv_id) is None:
        return jsonify({"error": "not found"}), 404
    share_id = get_active_share_id(username, conv_id)
    if not share_id:
        return jsonify({"shared": False})
    return jsonify({"shared": True, "share_id": share_id, "share_url": _share_url_for(share_id)})


@app.route("/api/conversations/<conv_id>/share", methods=["POST"])
@login_required
def api_create_share(conv_id):
    """Creates (or reuses an existing) public share link for this
    conversation — anyone with the link can view a read-only copy, no
    login required."""
    username = current_username()
    conv = load_conversation(username, conv_id)
    if conv is None:
        return jsonify({"error": "not found"}), 404
    if not conv.get("messages"):
        return jsonify({"error": "This chat has no messages yet — nothing to share."}), 400
    share_id = create_share_link(username, conv_id)
    return jsonify({"status": "shared", "share_id": share_id, "share_url": _share_url_for(share_id)})


@app.route("/api/conversations/<conv_id>/share", methods=["DELETE"])
@login_required
def api_revoke_share(conv_id):
    """Revokes any active public share link for this conversation — the old
    link stops working immediately."""
    username = current_username()
    revoke_share_link(username, conv_id)
    return jsonify({"status": "revoked"})


@app.route("/api/share/<share_id>", methods=["GET"])
def api_public_share(share_id):
    """Public, unauthenticated endpoint — returns a read-only view of a
    shared conversation. No user identity is exposed, only the messages."""
    ref = resolve_share_link(share_id)
    if not ref:
        return jsonify({"error": "This share link is invalid or has been revoked."}), 404
    conv = load_conversation(ref["username"], ref["conv_id"])
    if conv is None:
        return jsonify({"error": "This shared chat is no longer available."}), 404
    simplified = []
    for m in conv.get("messages", []):
        role = "user" if m["role"] == "user" else "ai"
        text_parts = [p.get("text", "") for p in m.get("parts", []) if "text" in p]
        entry = {"role": role, "text": "".join(text_parts)}
        if m.get("attachment_meta"):
            entry["attachment"] = m["attachment_meta"]
        simplified.append(entry)
    return jsonify({"title": conv.get("title", "Shared chat"), "messages": simplified})


@app.route("/api/share/<share_id>/continue", methods=["POST"])
@login_required
def api_continue_shared_chat(share_id):
    """Lets a visitor fork their own private, editable copy of a shared
    conversation so they can keep chatting. This NEVER touches the original
    owner's conversation — it copies the messages into a brand-new
    conversation under the visitor's own (anonymous, cookie-based) session,
    same as the existing "Duplicate" feature."""
    ref = resolve_share_link(share_id)
    if not ref:
        return jsonify({"error": "This share link is invalid or has been revoked."}), 404
    source_conv = load_conversation(ref["username"], ref["conv_id"])
    if source_conv is None:
        return jsonify({"error": "This shared chat is no longer available."}), 404

    visitor_username = current_username()
    new_id = str(uuid.uuid4())
    new_conv = {
        "title": (source_conv.get("title") or "Shared chat").strip(),
        "title_is_custom": True,
        "messages": json.loads(json.dumps(source_conv.get("messages", []))),  # deep copy
        "folder": None,
        "pinned": False,
        "archived": False,
    }
    save_conversation(visitor_username, new_id, new_conv)
    return jsonify({"status": "forked", "conversation_id": new_id})


@app.route("/share/<share_id>")
def public_share_page(share_id):
    """Public, unauthenticated read-only page rendering a shared chat —
    intentionally a separate, minimal template with no sidebar and no
    access to the viewer's own conversations. It does offer a "Continue
    this conversation" action, which forks a private copy for the visitor
    (see api_continue_shared_chat) rather than editing the original."""
    return Response(SHARE_PAGE, mimetype="text/html; charset=utf-8")


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
        "Below is a snippet of a chat conversation. Write ONE short, specific "
        "title (1-3 words) describing what the conversation is actually "
        "ABOUT — not a description of your task, not the words 'chat', "
        "'title', or 'conversation' themselves. No quotes, no punctuation "
        "at the end, plain text only. Prefer a simple category-style label "
        "when that fits.\n\n"
        "Examples:\n"
        "- If the user says 'hi' and gets a greeting back → 'Greeting'\n"
        "- If they ask about Python loops → 'Python Loops'\n"
        "- If they discuss weekend plans → 'Weekend Plans'\n\n"
        f"Conversation:\n{excerpt}\n\nTitle:"
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
    raw_title = re.sub(r'^Title:\s*', '', raw_title, flags=re.IGNORECASE).strip()

    # Safety filter: reject titles that describe the TASK instead of the
    # conversation's actual content (the model sometimes echoes back
    # phrases like "Generating Chat Titles" or "Chat Title" for very short/
    # generic exchanges like a bare "hi"). Fall back to a sensible default
    # derived from the first user message instead of saving junk.
    _meta_phrases = ("generating chat title", "chat title", "conversation title",
                      "title generation", "generate title", "untitled")
    if not raw_title or any(p in raw_title.lower() for p in _meta_phrases):
        first_user_text = ""
        for m in messages:
            if m.get("role") == "user":
                first_user_text = "".join(p.get("text", "") for p in m.get("parts", []) if "text" in p)
                break
        raw_title = (first_user_text[:40].strip() or "New chat")

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
        "Generate a short, natural chat title (1-4 words, no quotes, no punctuation "
        "at the end, no emoji, title case) that summarizes what this SPECIFIC "
        "conversation is about. Do NOT describe the task of generating a title — "
        "write about the actual topic discussed. Prefer a simple category-style "
        "label when that fits — for a simple greeting like 'hi'/'hello', use "
        "'Greeting'. Reply with ONLY the title text, nothing else.\n\n"
        f"User: {user_msg[:400]}\n"
        f"Assistant: {ai_reply[:400]}"
    )
    messages = [
        {"role": "system", "content": "You generate concise chat titles. Reply with only the title, no extra text."},
        {"role": "user", "content": prompt},
    ]
    result = _quick_completion(messages, api_key_groq, api_key_cerebras, max_tokens=16)
    if result:
        title = result.strip()
        title = title.strip('"\'“”‘’').strip()
        title = title.split("\n")[0].strip()
        title = re.sub(r'^\[.*?\]\s*', '', title).strip()  # strip any leaked [Instructions: ...] / bracketed prefix
        title = re.sub(r'[.!?]+$', '', title).strip()
        _meta_phrases = ("generating chat title", "chat title", "conversation title",
                          "title generation", "generate title", "untitled")
        looks_bad = (
            len(title) < 3 or
            any(ch in title for ch in '[]"“”') or
            title.lower().startswith(('based on', 'here is', 'here\'s', 'sure,', 'title:')) or
            any(p in title.lower() for p in _meta_phrases)
        )
        if title and not looks_bad:
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
    supabase_status = {"configured": bool(SUPABASE_URL and SUPABASE_KEY)}
    if supabase_status["configured"]:
        try:
            r = requests.get(sb("conversations?limit=1"), headers=sb_headers(), timeout=8)
            supabase_status["reachable"] = r.status_code == 200
            supabase_status["status_code"] = r.status_code
            if r.status_code != 200:
                supabase_status["error"] = r.text[:300]
        except Exception as e:
            supabase_status["reachable"] = False
            supabase_status["error"] = str(e)

        # A GET succeeding only proves the API key + table exist for reads.
        # The actual bug people hit is on INSERT (missing column, wrong
        # type, RLS blocking writes) — so actually try writing and deleting
        # a throwaway row and report the exact PostgREST error if it fails.
        if supabase_status.get("reachable"):
            test_id = "healthcheck-" + uuid.uuid4().hex[:8]
            try:
                w = requests.post(
                    sb("conversations"),
                    headers={**sb_headers(), "Prefer": "return=minimal"},
                    json={
                        "id": test_id, "username": "healthcheck", "title": "healthcheck",
                        "updated_at": time.time(), "messages": [],
                        "folder": None, "pinned": False, "archived": False,
                    },
                    timeout=10,
                )
                supabase_status["write_test"] = {
                    "status_code": w.status_code,
                    "success": w.status_code in (200, 201, 204),
                }
                if w.status_code not in (200, 201, 204):
                    supabase_status["write_test"]["error"] = w.text[:500]
                else:
                    requests.delete(
                        sb(f"conversations?id=eq.{test_id}"),
                        headers=sb_headers(), timeout=10,
                    )
            except Exception as e:
                supabase_status["write_test"] = {"success": False, "error": str(e)}

    return jsonify({
        "app": "Mythic AI",
        "serverless": IS_SERVERLESS,
        "providers": {
            "groq":     {"configured": bool(GROQ_API_KEY),     "model": GROQ_MODEL},
            "cerebras": {"configured": bool(CEREBRAS_API_KEY), "model": CEREBRAS_MODEL},
        },
        "storage": {
            "supabase": supabase_status,
            "using": "supabase" if supabase_status["configured"] else "local_json_files",
            "note": ("If 'configured' is true but 'reachable' is false or status_code isn't 200, "
                     "check that the 'conversations' table exists with the right columns, and that "
                     "SUPABASE_KEY is your service-role/secret key with insert/select access — see "
                     "the 'error' field above for the exact Supabase response.") if supabase_status["configured"] else
                    ("No SUPABASE_URL set — conversations are saved to local disk on the server. "
                     "On Render this works fine while the instance stays up, but is wiped on redeploy "
                     "and isn't shared across instances."),
            "shares_note": ("Share links (/share/<id>) use a separate `shares` table in Supabase when "
                             "configured: columns id (text, primary key), username (text), conv_id (text), "
                             "revoked (bool, default false), created_at (float8). If that table doesn't "
                             "exist yet, share links automatically fall back to local storage instead of "
                             "breaking, but they'll be lost on redeploy — create the table for persistence."),
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


# ── "Paste URL" helpers: ordinary webpages + flipbook viewers ────────────────

# Known flipbook/page-turn viewer platforms — if the URL's host matches one
# of these, it's almost certainly a JS canvas/image viewer with no real text
# in the page source, so we go straight to the embedded-PDF search / OCR path
# instead of trying (and failing) to scrape "readable text" from the shell.
_FLIPBOOK_DOMAINS = (
    "flippingbook.com", "issuu.com", "yumpu.com", "calameo.com",
    "anyflip.com", "joomag.com", "publitas.com", "heyzine.com",
    "flipsnack.com", "fliphtml5.com", "mmdigital.co.in",
)
# Telltale strings that show up in a flipbook viewer's HTML/JS even on
# self-hosted / white-labelled platforms we don't know by domain.
_FLIPBOOK_HTML_SIGNATURES = (
    "flipbook", "flip-book", "df-page", "page-flip", "pageflip",
    "turn.js", "df-parent", "flippingbook",
)


def _looks_like_flipbook(url: str, html: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    if any(d in host for d in _FLIPBOOK_DOMAINS):
        return True
    sample = (html or "")[:20000].lower()
    return any(sig in sample for sig in _FLIPBOOK_HTML_SIGNATURES)


def _find_embedded_document_link(html: str, base_url: str):
    """Many flipbook widgets (and some webpages) are generated from an
    uploaded PDF/DOCX that's still linked somewhere in the HTML or JS config
    — e.g. a hidden 'download original' link, or a `"pdfUrl": "...pdf"` entry
    in an embedded script. Grabbing that directly is far more reliable than
    OCR, so we check for it first. Returns an absolute URL or None."""
    if not html:
        return None
    candidates = []
    # href="....pdf" / .docx / .txt (typical <a> download links)
    for m in re.finditer(r'''href=["']([^"']+\.(?:pdf|docx|txt))(?:[?"'][^"']*)?["']''', html, re.IGNORECASE):
        candidates.append(m.group(1))
    # "file": "....pdf" / "pdfUrl": "....pdf" style JS config values
    for m in re.finditer(r'''["'](?:pdf|pdfUrl|file|source|bookFile|fileUrl)["']\s*:\s*["']([^"']+\.(?:pdf|docx|txt))["']''', html, re.IGNORECASE):
        candidates.append(m.group(1))
    for c in candidates:
        absolute = urllib.parse.urljoin(base_url, c)
        if absolute.lower().split("?")[0].endswith((".pdf", ".docx", ".txt")):
            return absolute
    return None


def _extract_readable_webpage_text(html: str) -> str:
    """Pulls readable body text out of an ordinary webpage, stripping nav,
    scripts, styles, and other chrome. Uses BeautifulSoup when available;
    falls back to a crude regex tag-strip so this still degrades gracefully
    if bs4 isn't installed."""
    if _BS4_AVAILABLE:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _webpage_title(html: str, fallback: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html or "")
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        if title:
            return title
    return fallback


def _resolve_basic_html_index(url: str, html: str):
    """FlipBuilder-family flipbook software ('Flip PDF', and similar clones)
    publishes an accessibility/SEO fallback under a 'basic-html' path
    alongside its JS page-turn viewer. Per FlipBuilder's own documentation,
    that fallback is generated by extracting the *real text* of the source
    PDF for search engines — so it's real readable text, not just images.
    Returns the absolute URL of that fallback's index page, or None."""
    if "basic-html" in url.lower():
        return url  # the user may have pasted this fallback URL directly
    m = re.search(r'''href=["']([^"']*basic-html/index\.html[^"']*)["']''', html, re.IGNORECASE)
    if m:
        return urllib.parse.urljoin(url, m.group(1))
    return None


def _extract_flipbuilder_basic_html_text(index_url: str, max_pages: int = 300):
    """Reads a FlipBuilder-style 'basic-html' fallback page by page. Returns
    (text, note) on success, or (None, reason) if it turns out to be
    image-only after all (some publishers disable the text-extraction option)."""
    try:
        r = requests.get(index_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except requests.RequestException as e:
        return None, f"Couldn't reach the basic-HTML version of this flipbook: {e}"

    html = r.text
    found = re.findall(r'''href=["']([^"']*/page(\d+)\.html)["']''', html, re.IGNORECASE)
    if not found:
        return None, "This looks like a Flip PDF/FlipBuilder-style flipbook, but no basic-HTML pages were found."
    by_num = {}
    for frag, num in found:
        by_num.setdefault(int(num), frag)
    ordered_frags = [by_num[n] for n in sorted(by_num)][:max_pages]

    texts = []
    for frag in ordered_frags:
        page_url = urllib.parse.urljoin(index_url, frag)
        try:
            pr = requests.get(page_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            pr.raise_for_status()
        except requests.RequestException:
            continue
        page_text = _extract_readable_webpage_text(pr.text).strip()
        if page_text:
            texts.append(page_text)

    full_text = "\n\n".join(texts).strip()
    if not full_text:
        return None, ("Found this flipbook's basic-HTML pages, but they contained no extractable "
                       "text (this publisher likely disabled text-extraction, leaving image-only pages).")
    note = f"Read {len(texts)} page(s) from this flipbook's built-in accessible text version (no OCR needed)."
    return full_text, note


def _extract_flipbook_via_ocr(url: str, max_pages: int = 40):
    """Best-effort flipbook reader: opens the page in a headless browser,
    clicks through it page by page, screenshots the viewer, and OCRs each
    screenshot. This is inherently fragile — every flipbook platform uses a
    different DOM/selector for its "next page" control — so it tries a list
    of common selectors and stops early if none of them seem to advance the
    book. Returns (text, note) or (None, error_message).

    Requires: playwright (+ `playwright install chromium`), pytesseract,
    and the tesseract-ocr binary installed on the host. See the import block
    near the top of this file for install instructions."""
    if not _FLIPBOOK_OCR_AVAILABLE:
        missing = []
        if not _PLAYWRIGHT_AVAILABLE:
            missing.append("playwright (`pip install playwright` + `playwright install chromium`)")
        if not _OCR_AVAILABLE:
            missing.append("pytesseract (`pip install pytesseract`) + the tesseract-ocr system package")
        if not _WATERMARK_AVAILABLE:
            missing.append("Pillow (`pip install Pillow`)")
        return None, ("This looks like a flipbook viewer. Reading it requires OCR support that "
                       "isn't installed on this server yet. Missing: " + "; ".join(missing) + ".")

    # Common "next page" selectors across popular flipbook platforms —
    # tried in order; the first one that's clickable AND changes the page
    # content wins. Add more as you encounter new platforms.
    NEXT_SELECTORS = [
        ".df-next-page", ".flipbook-next", ".next-page", "[aria-label='Next page']",
        "[aria-label='next']", ".turn-page-next", "button.next", ".df-arrow-right",
    ]
    texts = []
    try:
        with sync_playwright() as p:
            # --no-sandbox: Chromium refuses to run as root without this, and
            # most container platforms (including Render's Docker runtime) run
            # as root by default. --disable-dev-shm-usage avoids crashes from
            # Docker's small default /dev/shm size.
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            page.goto(url, timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(2000)  # let the viewer finish initial render

            next_selector = None
            for sel in NEXT_SELECTORS:
                if page.locator(sel).count() > 0:
                    next_selector = sel
                    break

            seen_hashes = set()
            for i in range(max_pages):
                shot = page.screenshot()
                img_hash = hashlib.md5(shot).hexdigest()
                if img_hash in seen_hashes:
                    break  # page didn't change — we've reached the end or got stuck
                seen_hashes.add(img_hash)
                try:
                    from io import BytesIO
                    page_text = pytesseract.image_to_string(Image.open(BytesIO(shot)))
                except Exception:
                    page_text = ""
                if page_text.strip():
                    texts.append(page_text.strip())
                if not next_selector:
                    break  # can't advance — only OCR the first visible page/spread
                try:
                    page.click(next_selector, timeout=3000)
                    page.wait_for_timeout(900)
                except Exception:
                    break
            browser.close()
    except Exception as e:
        return None, f"Flipbook OCR failed: {e}"

    full_text = "\n\n".join(texts).strip()
    if not full_text:
        return None, ("Couldn't read any text from this flipbook — OCR ran but found nothing "
                       "recognizable. The page images may be too low-resolution or stylized to OCR.")
    note = f"Read via OCR from a flipbook viewer ({len(texts)} page(s) scanned — may be incomplete)."
    return full_text, note


@app.route("/api/fetch-url-document", methods=["POST"])
@login_required
def api_fetch_url_document():
    """Pulls readable text from whatever kind of 'book URL' the user pastes:
      1. A direct PDF/DOCX/TXT download link       -> parsed like a file upload
      2. An ordinary webpage (article, story, etc.) -> readable text scraped from the HTML
      3. A flipbook-style viewer (FlippingBook, Issuu, Yumpu, mmdigital, ...)
           a. first tries to find an embedded/original PDF link on the page
           b. falls back to headless-browser + OCR page-by-page (if configured)
    Caps document downloads to DOCUMENT_UPLOAD_BYTES, streaming so we abort
    early instead of reading an arbitrarily large body first."""
    DOC_EXTENSIONS = (".pdf", ".docx", ".doc", ".txt")
    DOC_CONTENT_TYPES = (
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    )

    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "Please enter a valid http:// or https:// URL"}), 400

    def _download_and_extract_doc(doc_url, timeout=20):
        """Shared path for downloading + parsing a direct PDF/DOCX/TXT link."""
        r = requests.get(doc_url, timeout=timeout, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "").split(";")[0].strip()
        raw = bytearray()
        for chunk in r.iter_content(chunk_size=65536):
            raw.extend(chunk)
            if len(raw) > DOCUMENT_UPLOAD_BYTES:
                limit_mb = DOCUMENT_UPLOAD_BYTES // (1024 * 1024)
                raise ValueError(f"That file is larger than {limit_mb}MB — please download it and upload it directly instead.")
        fname = (doc_url.split("/")[-1].split("?")[0] or "document").strip() or "document"
        txt, note = extract_text_from_attachment(fname, ct, bytes(raw))
        return txt, note, fname, ct

    path = urllib.parse.urlparse(url).path.lower()

    # ── Case 1: direct document link ─────────────────────────────────────
    if path.endswith(DOC_EXTENSIONS):
        try:
            text, note, filename, content_type = _download_and_extract_doc(url)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except requests.RequestException:
            return jsonify({"error": "Could not reach that link — please double-check it's a direct, "
                                      "publicly accessible file link and try again."}), 502
        if content_type and content_type not in DOC_CONTENT_TYPES:
            return jsonify({"error": "That link didn't actually return a PDF/DOCX/TXT file "
                                      "(the server sent something else back)."}), 400
        if text is None:
            return jsonify({"error": "Couldn't read that as a PDF, DOCX, or TXT file."}), 400
        if not text.strip():
            return jsonify({"error": note or "No readable text was found in that document."}), 400
        return jsonify({"text": text, "note": note, "filename": filename})

    # ── Fetch the page as HTML so we can figure out what we're looking at ──
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException:
        return jsonify({"error": "Could not reach that link — please double-check the URL and try again."}), 502

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()

    # A no-extension URL might still redirect straight to a real document.
    if content_type in DOC_CONTENT_TYPES:
        fname = (url.split("/")[-1].split("?")[0] or "document").strip() or "document"
        text, note = extract_text_from_attachment(fname, content_type, resp.content)
        if text and text.strip():
            return jsonify({"text": text, "note": note, "filename": fname})

    html = resp.text

    # ── Case 2/3: figure out if this is a flipbook, and try the embedded-PDF shortcut either way ──
    embedded_doc_url = _find_embedded_document_link(html, url)
    if embedded_doc_url:
        try:
            text, note, filename, doc_ct = _download_and_extract_doc(embedded_doc_url)
            if text and text.strip() and (not doc_ct or doc_ct in DOC_CONTENT_TYPES):
                note = (note + " " if note else "") + "(Found the original document embedded in that page.)"
                return jsonify({"text": text, "note": note.strip(), "filename": filename})
        except Exception:
            pass  # fall through to OCR / webpage scraping below

    if _looks_like_flipbook(url, html):
        text, note_or_error = _extract_flipbook_via_ocr(url)
        if text:
            title = _webpage_title(html, "flipbook")
            return jsonify({"text": text, "note": note_or_error, "filename": f"{title}.txt"})
        return jsonify({"error": note_or_error}), 501 if not _FLIPBOOK_OCR_AVAILABLE else 502

    # ── Case 2: ordinary webpage ─────────────────────────────────────────
    text = _extract_readable_webpage_text(html)
    if not text.strip():
        return jsonify({"error": "That page didn't have any readable text to extract."}), 400
    title = _webpage_title(html, url.split("/")[-1] or "page")
    return jsonify({"text": text, "note": "Extracted readable text from a webpage.", "filename": f"{title}.txt"})


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
        att_mime = (attachment.get("mimeType") or "")
        att_name = (attachment.get("name") or "").lower()
        is_document = (att_mime == "application/pdf" or att_name.endswith((".pdf", ".docx", ".txt", ".md")))
        effective_limit = DOCUMENT_UPLOAD_BYTES if is_document else MAX_UPLOAD_BYTES
        if len(raw) > effective_limit:
            limit_mb = effective_limit // (1024 * 1024)
            return jsonify({"error": f"attachment too large (max {limit_mb}MB for this file type)"}), 400

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


# ══════════════════════════════════════════════════════════════════════════
# NEW FEATURE: real, sandboxed code execution (Python/JS/etc.) for the Code
# tab and for "Run" on non-HTML Artifacts.
#
# Execution happens on Piston (emkc.org's free public code-execution API),
# NOT on this server. That matters: this Render instance holds real secrets
# (GROQ/CEREBRAS/SUPABASE keys) in its environment, so running arbitrary,
# untrusted code IN THIS PROCESS would let anyone with the invite link read
# those secrets or pivot into this box. Piston runs the code in its own
# disposable sandbox instead, so a malicious snippet can at worst mess with
# a throwaway container that has none of this app's secrets.
# Still gated behind the existing VIP session flag (server-side, not just a
# client-side lock icon) — it's real third-party compute, not something to
# leave wide open to every anonymous visitor of the public invite link.
# ══════════════════════════════════════════════════════════════════════════
PISTON_API = "https://emkc.org/api/v2/piston/execute"
PISTON_RUNTIMES_API = "https://emkc.org/api/v2/piston/runtimes"
_piston_runtimes_cache = {"ts": 0.0, "data": []}
_PISTON_LANGUAGE_ALIASES = {
    "py": "python", "python3": "python", "js": "javascript", "node": "javascript",
    "nodejs": "javascript", "ts": "typescript", "sh": "bash", "shell": "bash",
    "c++": "cpp", "golang": "go",
}
_PISTON_FILENAMES = {
    "python": "main.py", "javascript": "main.js", "typescript": "main.ts",
    "bash": "main.sh", "c": "main.c", "cpp": "main.cpp", "java": "Main.java",
    "go": "main.go", "rust": "main.rs", "ruby": "main.rb", "php": "main.php",
}

def _piston_runtimes():
    now = time.time()
    if _piston_runtimes_cache["data"] and (now - _piston_runtimes_cache["ts"]) < 3600:
        return _piston_runtimes_cache["data"]
    try:
        r = requests.get(PISTON_RUNTIMES_API, timeout=10)
        if r.status_code == 200:
            _piston_runtimes_cache["data"] = r.json()
            _piston_runtimes_cache["ts"] = now
    except Exception as e:
        print(f"[Piston] failed to fetch runtimes: {e}")
    return _piston_runtimes_cache["data"]


@app.route("/api/execute-code", methods=["POST"])
@login_required
def api_execute_code():
    if not session.get("vip_unlocked"):
        return jsonify({"error": "Unlock VIP mode first to run code."}), 403
    data = request.get_json(force=True) or {}
    code = (data.get("code") or "")
    if not code.strip():
        return jsonify({"error": "No code provided."}), 400
    if len(code) > 20000:
        return jsonify({"error": "Code is too long (max 20,000 characters)."}), 400
    lang_in = (data.get("language") or "python").strip().lower()
    lang = _PISTON_LANGUAGE_ALIASES.get(lang_in, lang_in)

    runtimes = _piston_runtimes()
    version = "*"
    if runtimes:
        match = next((r for r in runtimes if r.get("language") == lang
                       or lang in (r.get("aliases") or [])), None)
        if not match:
            supported = sorted({r.get("language") for r in runtimes})
            return jsonify({"error": f"'{lang_in}' isn't a supported language.",
                             "supported": supported[:30]}), 400
        version = match.get("version", "*")

    filename = _PISTON_FILENAMES.get(lang, "main.txt")
    try:
        resp = requests.post(PISTON_API, json={
            "language": lang,
            "version": version,
            "files": [{"name": filename, "content": code}],
            "stdin": (data.get("stdin") or "")[:5000],
            "run_timeout": 8000,
            "compile_timeout": 10000,
        }, timeout=25)
        if resp.status_code != 200:
            return jsonify({"error": f"Execution service returned HTTP {resp.status_code}."}), 502
        result = resp.json()
        run = result.get("run") or {}
        compiled = result.get("compile") or {}
        return jsonify({
            "stdout": run.get("stdout", ""),
            "stderr": run.get("stderr", "") or compiled.get("stderr", ""),
            "exit_code": run.get("code"),
            "language": lang,
            "version": version,
        })
    except requests.exceptions.Timeout:
        return jsonify({"error": "Execution timed out."}), 504
    except Exception as e:
        return jsonify({"error": f"Execution failed: {e}"}), 500


# ══════════════════════════════════════════════════════════════════════════
# NEW FEATURE: Cowork mode — real multi-step task execution, not just a
# different system prompt. Breaks the task into steps, works through each
# one (carrying forward what earlier steps produced), then synthesizes a
# single final answer. Uses the existing _quick_completion() Groq→Cerebras
# fallback helper, so it inherits the same silent-provider-fallback behavior
# as the rest of the app.
# ══════════════════════════════════════════════════════════════════════════
@app.route("/api/cowork/run", methods=["POST"])
@login_required
def api_cowork_run():
    if not session.get("vip_unlocked"):
        return jsonify({"error": "Unlock VIP mode first to use Cowork."}), 403
    data = request.get_json(force=True) or {}
    task = (data.get("task") or "").strip()
    if not task:
        return jsonify({"error": "No task provided."}), 400
    task = task[:4000]
    user_groq = (data.get("groqKey") or "").strip() or None
    user_cerebras = (data.get("cerebrasKey") or "").strip() or None

    plan_text = _quick_completion(
        [
            {"role": "system", "content": (
                "You are a meticulous project planner. Break the user's task into 3 to 6 "
                "short, concrete, sequential steps a capable assistant could execute one "
                "at a time. Reply with ONLY a numbered list, one short step per line, "
                "nothing else — no preamble, no explanation."
            )},
            {"role": "user", "content": task},
        ],
        user_groq, user_cerebras, max_tokens=300,
    ) or ""
    steps = []
    for line in plan_text.splitlines():
        line = re.sub(r"^\s*[\d]+[\.\)]\s*", "", line).strip("-• \t")
        if line:
            steps.append(line)
    steps = steps[:6] or [task]

    step_results = []
    running_context = ""
    for i, step in enumerate(steps, 1):
        step_prompt = (
            f"Overall task: {task}\n\n"
            f"Progress so far:\n{running_context[-3000:] or '(nothing yet)'}\n\n"
            f"Do ONLY this step ({i} of {len(steps)}): {step}\n"
            "Give a direct, concrete result for this step alone — don't restate the "
            "whole task or repeat earlier steps."
        )
        result = _quick_completion(
            [
                {"role": "system", "content": "You are a focused task executor. Do exactly "
                                               "the step given, concisely and concretely."},
                {"role": "user", "content": step_prompt},
            ],
            user_groq, user_cerebras, max_tokens=600,
        ) or "(no result — provider unavailable)"
        step_results.append({"step": step, "result": result})
        running_context += f"\nStep {i} — {step}:\n{result}\n"

    final_answer = _quick_completion(
        [
            {"role": "system", "content": "You synthesize completed work into one final, "
                                           "coherent answer or deliverable."},
            {"role": "user", "content": (
                f"Task: {task}\n\nStep-by-step work completed:\n{running_context[-6000:]}\n\n"
                "Write the final, complete answer for the task, weaving the step results "
                "together into one coherent response. Don't just list the steps again."
            )},
        ],
        user_groq, user_cerebras, max_tokens=1400,
    ) or running_context

    return jsonify({"task": task, "steps": step_results, "final_answer": final_answer})


# ══════════════════════════════════════════════════════════════════════════
# NEW FEATURE: search across every one of this visitor's conversations
# (not just the currently open one).
# ══════════════════════════════════════════════════════════════════════════
@app.route("/api/search", methods=["GET"])
@login_required
def api_search_messages():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"results": [], "query": q})
    q_lower = q.lower()
    username = current_username()
    results = []
    for c in list_conversations(username):
        conv = load_conversation(username, c["id"])
        if not conv:
            continue
        for idx, m in enumerate(conv.get("messages", []) or []):
            text = m.get("text") or ""
            pos = text.lower().find(q_lower)
            if pos == -1:
                continue
            start = max(0, pos - 40)
            snippet = text[start:start + 160].strip()
            results.append({
                "conv_id": c["id"],
                "title": c.get("title") or "Untitled",
                "role": m.get("role"),
                "msg_index": idx,
                "snippet": ("…" if start > 0 else "") + snippet + ("…" if start + 160 < len(text) else ""),
            })
            if len(results) >= 50:
                break
        if len(results) >= 50:
            break
    return jsonify({"results": results, "query": q})


# ══════════════════════════════════════════════════════════════════════════
# NEW FEATURE: reminders / scheduled tasks. Reuses the existing Web Push
# infrastructure (send_push_notification_to_user, built for re-engagement
# notifications) to actually deliver the reminder at the right time.
# ══════════════════════════════════════════════════════════════════════════
_REMINDERS_FILE = _os.path.join(_DATA_DIR, "reminders.json")
_reminders_lock = threading.Lock()

def _load_reminders():
    with _reminders_lock:
        if _os.path.exists(_REMINDERS_FILE):
            try:
                with open(_REMINDERS_FILE, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

def _save_reminders(data):
    with _reminders_lock:
        try:
            with open(_REMINDERS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[reminders] failed to save: {e}")


@app.route("/api/reminders", methods=["GET"])
@login_required
def api_list_reminders():
    username = current_username()
    all_r = _load_reminders()
    mine = [r for r in all_r.values() if r.get("username") == username and not r.get("fired")]
    mine.sort(key=lambda r: r.get("fire_at", 0))
    return jsonify({"reminders": mine})


@app.route("/api/reminders", methods=["POST"])
@login_required
def api_create_reminder():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()[:200]
    fire_at = data.get("fire_at")
    if not text or fire_at is None:
        return jsonify({"error": "'text' and 'fire_at' (unix timestamp, seconds) are required."}), 400
    try:
        fire_at = float(fire_at)
    except (TypeError, ValueError):
        return jsonify({"error": "'fire_at' must be a numeric unix timestamp."}), 400
    if fire_at < time.time() - 60:
        return jsonify({"error": "That time is in the past."}), 400
    rid = uuid.uuid4().hex[:12]
    all_r = _load_reminders()
    all_r[rid] = {
        "id": rid, "username": current_username(), "text": text,
        "fire_at": fire_at, "created_at": time.time(), "fired": False,
    }
    _save_reminders(all_r)
    return jsonify({"reminder": all_r[rid]})


@app.route("/api/reminders/<rid>", methods=["DELETE"])
@login_required
def api_delete_reminder(rid):
    all_r = _load_reminders()
    if rid in all_r and all_r[rid].get("username") == current_username():
        del all_r[rid]
        _save_reminders(all_r)
    return jsonify({"ok": True})


def _reminders_check_loop():
    """Every 30s, fires any reminder whose time has come, via Web Push.
    Render/always-on only, same limitation as the re-engagement loop below —
    Vercel's serverless functions freeze between requests so a background
    thread there would never actually get to run on schedule."""
    while True:
        try:
            all_r = _load_reminders()
            now = time.time()
            changed = False
            for rid, r in list(all_r.items()):
                if not r.get("fired") and r.get("fire_at", 0) <= now:
                    send_push_notification_to_user(
                        r.get("username", ""), "⏰ Reminder", r.get("text", ""), url="/",
                    )
                    r["fired"] = True
                    changed = True
            if changed:
                _save_reminders(all_r)
        except Exception as e:
            print(f"[reminders] loop error: {e}")
        time.sleep(30)


_reminders_thread_started = False

def _start_reminders_thread_once():
    global _reminders_thread_started
    if _reminders_thread_started or IS_SERVERLESS:
        return
    threading.Thread(target=_reminders_check_loop, daemon=True).start()
    _reminders_thread_started = True

_start_reminders_thread_once()


# ══════════════════════════════════════════════════════════════════════════
# NEW FEATURE: export ALL conversations as one downloadable .zip backup, and
# re-import them later (e.g. after clearing cookies, or on another device/
# browser — since accounts are anonymous per-browser-cookie, this is also
# the only way to move your chats to a different browser at all).
# ══════════════════════════════════════════════════════════════════════════
@app.route("/api/backup/export", methods=["GET"])
@login_required
def api_backup_export():
    import io as _io, zipfile as _zipfile
    username = current_username()
    buf = _io.BytesIO()
    manifest = []
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
        for c in list_conversations(username):
            conv = load_conversation(username, c["id"])
            if not conv:
                continue
            payload = dict(conv)
            payload["id"] = c["id"]
            zf.writestr(f"conversations/{c['id']}.json",
                        json.dumps(payload, ensure_ascii=False))
            manifest.append({"id": c["id"], "title": c.get("title")})
        zf.writestr("manifest.json", json.dumps({
            "exported_at": time.time(), "count": len(manifest), "conversations": manifest,
        }, ensure_ascii=False))
    buf.seek(0)
    return Response(buf.read(), mimetype="application/zip", headers={
        "Content-Disposition": f'attachment; filename="mythic-ai-backup-{int(time.time())}.zip"',
    })


@app.route("/api/backup/import", methods=["POST"])
@login_required
def api_backup_import():
    import io as _io, zipfile as _zipfile
    username = current_username()
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded."}), 400
    raw = f.read()
    if len(raw) > 25 * 1024 * 1024:
        return jsonify({"error": "Backup file too large (max 25MB)."}), 400
    try:
        zf = _zipfile.ZipFile(_io.BytesIO(raw))
    except Exception as e:
        return jsonify({"error": f"Not a valid backup file: {e}"}), 400
    imported = 0
    for name in zf.namelist():
        if not name.startswith("conversations/") or not name.endswith(".json"):
            continue
        try:
            payload = json.loads(zf.read(name).decode("utf-8"))
        except Exception:
            continue
        # Always import under a brand-new id, even if the backup file still
        # has its original id — prevents an import from silently clobbering
        # an existing chat that happens to share the same id.
        payload.pop("id", None)
        new_id = uuid.uuid4().hex[:12]
        save_conversation(username, new_id, payload)
        imported += 1
    return jsonify({"imported": imported})


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
