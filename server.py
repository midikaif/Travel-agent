import os
import tempfile
import uuid
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import numpy as np
import scipy.io.wavfile as wav
import redis

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# ── Config ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
REDIS_URL = ""
# os.getenv("REDIS_URL", "")

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "TX3LPaxmHKxFdv7VOQHJ")

# ── Constants ───────────────────────────────────────────

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

llm = Groq(api_key=GROQ_API_KEY)
lab_tts = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# ── App ─────────────────────────────────────────────────
app = FastAPI(title="ramesh Voice Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-Text", "X-Agent-Text"],
)


# ── Helpers ─────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio format (webm, ogg, mp4) to WAV 16 kHz mono.
    Browser MediaRecorder sends webm - Whisper needs WAV. Falls back gracefully if ffmpeg not available.
    """
    output_path = input_path.replace(".tmp", ".wav") + ".wav"
    ret = os.system(f'ffmpeg -y -i "{input_path}" -ar 16000 -ac 1 -f wav "{output_path}" -loglevel quiet')

    if ret == 0  and os.path.exists(output_path):
        return output_path
    # ffmpeg not available - return original and let Whisper try
    return input_path

def amplify_audio(audio_path: str) -> str:
    """Boost low-volume WAV recordings before Whisper."""
    try:
        rate, data = wav.read(audio_path)
        max_amp = np.abs(data).max()
        if 0 < max_amp < 10000:
            data = (data * (20000 / max_amp)).astype(np.int16)
            wav.write(audio_path, rate, data)
    except Exception:
        pass
    return audio_path


def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = llm.audio.transcriptions.create(
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
    response = llm.chat.completions.create(
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

    # if ELEVENLABS_API_KEY:
    try:
        audio = lab_tts.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        save(audio, tmp.name)
    except Exception as err:
        #Fallback to gTTS if no Elevenlabs key
        from gtts import gTTS
        tts = gTTS(text=text, lang="hi", slow=False)
        tts.save(tmp.name)
    
    return tmp.name


# ── Routes ──────────────────────────────────────────────

@app.get("/demo", response_class=HTMLResponse)
def demo():
    with open("travel-agent-landing-page.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/")
def root():
    tts_engine = "ElevenLabs" if ELEVENLABS_API_KEY else "gTTS (fallback)"
    return {"status": "ramesh is online 🚌", "tts_engine": tts_engine}


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
    Flow: audio -> covert to WAV -> amplify -> Whisper STT -> Llama LLM -> ElevenLabs TTs/gTTS -> MP3
    """
    # 1. Save uploaded audio to temp file
    audio_bytes = await audio.read()
    
    suffix = ".webm" if "webm" in (audio.content_type or "") else ".wav"
    tmp_raw = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_raw.write(audio_bytes)
    tmp_raw.close()

    tmp_wav = None
    mp3_path = None

    try:

        tmp_wav = convert_to_wav(tmp_raw.name)
        amplify_audio(tmp_wav)
        
        #Transcribe
        user_text = transcribe(tmp_wav)
        if not user_text:
            raise HTTPException(status_code=400, detail="Awaaz nahi aayi — please dobara bolo")

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
                "Access-Control_Expose-Headers":
                "X-User-Text, X-Agent_Text",
            },
           
        )

    finally:
        # Clean up temp files
        for path in [tmp_raw.name, tmp_wav]:
            if path and path != tmp_raw.name and os.path.exists(path):
                try: os.unlink(path)
                except: pass
        try: os.unlink(tmp_raw.name)
        except: pass


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
