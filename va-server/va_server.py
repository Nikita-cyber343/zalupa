#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 va_server.py — server local de inferență pentru asistentul vocal ESP32-S3
============================================================================
 Lanț: faster-whisper (STT) -> Ollama/llama3.2 (LLM) -> Piper (TTS)
 Rulează 100% local. Fără servicii cloud.

 NOU față de versiunea anterioară:
   - /voice întoarce, pe lângă audio, și textul (transcriere + răspuns)
     în anteturile HTTP  X-Transcript  și  X-Reply  (URL-encodate),
     ca interfața web de pe placă să poată afișa istoricul conversației.
   - endpoint nou  /text  : primește o întrebare scrisă (din interfața web),
     rulează doar LLM + TTS (fără STT) și întoarce audio + text.

 Pornire:
   uvicorn va_server:app --host 0.0.0.0 --port 8000
============================================================================
"""

import io, wave, numpy as np
from urllib.parse import quote
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from faster_whisper import WhisperModel
from openai import OpenAI
from piper import PiperVoice

# ----------------------------------------------------------------- configurare
WHISPER_MODEL = "base"               # "tiny" transcrie greșit; "base" e minimul fiabil
OLLAMA_MODEL  = "llama3.2:1b"        # model rapid; 3b = mai bun, mai lent
OLLAMA_URL    = "http://localhost:11434/v1"
PIPER_VOICE   = "en_US-lessac-medium.onnx"
TTS_RATE_SRC  = 22050                # frecvența de ieșire a vocii Piper
TTS_RATE_DST  = 16000                # frecvența așteptată de placă
MAX_TOKENS    = 60                   # răspunsuri scurte = generare rapidă
SYSTEM_PROMPT = ("You are a concise voice assistant. Answer in one or two short "
                 "sentences, plainly, with no markdown, lists or emoji.")

# ----------------------------------------------------------------- inițializare
print("Încarc modelele...")
stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
llm = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
voice = PiperVoice.load(PIPER_VOICE)
app = FastAPI()
print("Server pregătit.")

# ------------------------------------------------------------------- funcții
def transcribe(wav_bytes: bytes) -> str:
    """Audio WAV -> text (STT)."""
    segments, _ = stt.transcribe(io.BytesIO(wav_bytes), language="en", beam_size=1)
    return " ".join(s.text for s in segments).strip()

def chat(text: str) -> str:
    """Text -> răspuns (LLM)."""
    r = llm.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": text}],
        temperature=0.5,
        max_tokens=MAX_TOKENS,
    )
    return r.choices[0].message.content.strip()

def synthesize(text: str) -> bytes:
    """Text -> audio WAV 16 kHz mono (TTS + reeșantionare)."""
    chunks = []
    for chunk in voice.synthesize_stream_raw(text):
        chunks.append(np.frombuffer(chunk, dtype=np.int16))
    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.int16)

    # reeșantionare 22050 -> 16000 prin interpolare liniară
    n_dst = int(len(audio) * TTS_RATE_DST / TTS_RATE_SRC)
    x_src = np.linspace(0, 1, len(audio), endpoint=False)
    x_dst = np.linspace(0, 1, n_dst, endpoint=False)
    audio = np.interp(x_dst, x_src, audio).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(TTS_RATE_DST)
        w.writeframes(audio.tobytes())
    return buf.getvalue()

def audio_response(wav: bytes, transcript: str, reply: str) -> Response:
    """Împachetează audio + text (în anteturi) într-un răspuns HTTP."""
    return Response(content=wav, media_type="audio/wav", headers={
        "X-Transcript": quote(transcript),   # URL-encodat; decodat în browser
        "X-Reply":      quote(reply),
    })

# ------------------------------------------------------------------- endpoints
@app.post("/voice")
async def voice(request: Request):
    """Interacțiune vocală: audio -> STT -> LLM -> TTS -> audio + text."""
    import time
    wav_in = await request.body()

    t0 = time.time(); text  = transcribe(wav_in); t1 = time.time()
    reply = chat(text);                            t2 = time.time()
    wav   = synthesize(reply);                     t3 = time.time()
    print(f"[STT {t1-t0:.1f}s] [LLM {t2-t1:.1f}s] [TTS {t3-t2:.1f}s]  "
          f"'{text}' -> '{reply}'")
    return audio_response(wav, text, reply)

@app.post("/text")
async def text_query(request: Request):
    """Interacțiune scrisă (din interfața web): text -> LLM -> TTS -> audio + text."""
    import time
    data  = await request.json()
    text  = (data.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    t0 = time.time(); reply = chat(text);     t1 = time.time()
    wav   = synthesize(reply);                t2 = time.time()
    print(f"[TEXT] [LLM {t1-t0:.1f}s] [TTS {t2-t1:.1f}s]  '{text}' -> '{reply}'")
    return audio_response(wav, text, reply)

@app.get("/health")
async def health():
    return {"status": "ok", "model": OLLAMA_MODEL}
