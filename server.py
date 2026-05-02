import os
import tempfile
import uuid
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from gtts import gTTS
import numpy as np
import scipy.io.wavfile as wav
import redis

# ── Config ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", None)

SYSTEM_PROMPT = """
Tu "ramesh" hai, ek friendly inbound travel ticketing agent.

LANGUAGE:
- Simple Hindi ya Hinglish mein baat kar.
- Sentences chhote aur natural rakh.
- Friendly, conversational tone use kar.
- Amount hamesha Hindi mein bol: 1000=ek hazaar, 1200=barah so,
  2000=do hazaar, 2500=pachchees so, 2300=teis so, 2100=ekkis so, 4500=chauntaalees so.

GOAL:
1) Pickup location poochh (valid: Kanpur, Orai, Jhansi)
2) Destination poochh (valid: Ahmedabad, Rajkot, Surat)
3) Number of passengers poochh
4) Route validate kar, nearby cities ke liye nearest hub suggest kar
5) Pricing: Ahmedabad=1000, Rajkot/Surat=1200. NOP>1 toh sleeper upsell.
   Female toh single sleeper: 2500->2300->2100->2000 (floor).
   NOP>2 double sleeper: 4500 for 4 pax.
6) Confirm karo. YES toh payment guide. NO toh ek baar convince.

ASR HANDLING:
- "Rahi/Orayi" = Orai, "Jansi/Jhasi" = Jhansi
- "Kanpore/Kanpoor" = Kanpur, "Amdabad/Anyaabad" = Ahmedabad
- Seedha reject mat karo, pehle confirm karo.
- Agar bilkul samajh nahi aaya: "Kya aap dobara bol sakte ho?"

CONSTRAINTS:
- 2-3 sentences max per response.
- Ek sawaal ek baar.
- Kabhi card/bank details mat maango.
"""

# ── Redis session store ──────────────────────────────────
# Connect to Upstash Redis if URL provided, else use in-memory fallback
if REDIS_URL:
    r = redis.from_url(REDIS_URL, decode_responses=True)
else:
    # Fallback for local development
    sessions: dict[str, list] = {}
    r = None

client = Groq(api_key=GROQ_API_KEY)

# ── App ─────────────────────────────────────────────────
app = FastAPI(title="ramesh Voice Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this when you go to production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────

def amplify_audio(audio: np.ndarray, target: int = 20000) -> np.ndarray:
    max_amp = np.abs(audio).max()
    if 0 < max_amp < 10000:
        audio = (audio * (target / max_amp)).astype(np.int16)
    return audio


def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(audio_path, f.read()),
            model="whisper-large-v3",
            language="hi",
            prompt="Yeh ek bus booking conversation hai. Cities: Kanpur, Orai, Jhansi, Ahmedabad, Rajkot, Surat.",
        )
    return result.text.strip()


def get_llm_response(session_id: str, user_text: str) -> str:
    # Get conversation history from Redis or memory
    if r:
        history_json = r.get(f"session:{session_id}")
        history = json.loads(history_json) if history_json else []
    else:
        history = sessions.get(session_id, [])

    # Add user message
    history.append({"role": "user", "content": user_text})

    # Get LLM response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        max_tokens=200,
    )

    # Add assistant response
    reply = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})

    # Save back to Redis or memory (TTL: 24 hours)
    if r:
        r.setex(f"session:{session_id}", 86400, json.dumps(history))
    else:
        sessions[session_id] = history

    return reply


def text_to_speech(text: str) -> str:
    """Convert text to speech, return path to MP3 file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    tts = gTTS(text=text, lang="hi", slow=False)
    tts.save(tmp.name)
    return tmp.name


# ── Routes ──────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ramesh is online 🚌"}


@app.get("/session")
def new_session():
    """Create a new conversation session. Call this when a new user opens the page."""
    session_id = str(uuid.uuid4())
    if r:
        r.setex(f"session:{session_id}", 86400, json.dumps([]))
    else:
        sessions[session_id] = []
    return {"session_id": session_id}


@app.post("/chat/{session_id}")
async def chat(session_id: str, audio: UploadFile = File(...)):
    """
    Main endpoint. Accepts audio file, returns MP3 response.
    Flow: audio -> Whisper STT -> Llama LLM -> gTTS -> MP3
    """
    # 1. Save uploaded audio to temp file
    audio_bytes = await audio.read()
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in.write(audio_bytes)
    tmp_in.close()

    try:
        # 2. Amplify if needed (handles low-volume mics)
        try:
            rate, data = wav.read(tmp_in.name)
            data = amplify_audio(data)
            wav.write(tmp_in.name, rate, data)
        except Exception:
            pass  # if amplification fails, send as-is

        # 3. Transcribe with Groq Whisper
        user_text = transcribe(tmp_in.name)
        if not user_text:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # 4. Get LLM response
        reply_text = get_llm_response(session_id, user_text)

        # 5. Convert reply to speech
        mp3_path = text_to_speech(reply_text)

        # 6. Return audio + transcription in headers for debugging
        return FileResponse(
            mp3_path,
            media_type="audio/mpeg",
            headers={
                "X-User-Text": user_text.encode("utf-8").decode("latin-1", errors="replace"),
                "X-Agent-Text": reply_text.encode("utf-8").decode("latin-1", errors="replace"),
            },
            background=None,
        )

    finally:
        os.unlink(tmp_in.name)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if r:
        r.delete(f"session:{session_id}")
    else:
        sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/greeting/{session_id}")
def greeting(session_id: str):
    """
    Get the opening greeting as audio.
    Call this when the page loads to play ramesh's intro.
    """
    text = "Namaste! Main ramesh hoon, aapka bus booking agent. Aap kahan se travel karna chahte ho?"
    greeting_history = [{"role": "assistant", "content": text}]
    if r:
        r.setex(f"session:{session_id}", 86400, json.dumps(greeting_history))
    else:
        sessions[session_id] = greeting_history
    mp3_path = text_to_speech(text)
    return FileResponse(mp3_path, media_type="audio/mpeg")
