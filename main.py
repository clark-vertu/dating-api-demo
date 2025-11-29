import base64
import os
import tempfile
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# 可配置模型，方便未来切换
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")
GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
GROQ_TTS_MODEL = os.environ.get("GROQ_TTS_MODEL", "playai-tts")

session_context: Dict[str, str] = {}  # key: session_id, value: last_assistant_text


def _normalize_text(s: str) -> str:
    """简单归一化：小写 + 去空白，用于粗略比较“是不是回声”"""
    return "".join(ch for ch in s.lower() if not ch.isspace())


@app.post("/api/voice_chat")
async def voice_chat(
    file: UploadFile = File(...),
    is_interruption: str = Form(default="false"),
    session_id: str = Form(default=""),
) -> Dict[str, str]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    filename = file.filename or "audio.m4a"

    # ---------- 1. STT ----------
    try:
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=GROQ_WHISPER_MODEL,
            response_format="json",
            temperature=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

    user_text = getattr(transcription, "text", None)
    print("user_text", user_text)

    if not user_text:
        raise HTTPException(status_code=500, detail="STT result has no text")

    is_interrupt = is_interruption.lower() == "true"

    # ---------- 1.5 打断场景下的快速过滤：太短 / 回声 ----------
    if is_interrupt:
        user_text_stripped = user_text.strip()
        if not user_text_stripped or len(user_text_stripped) < 3:
            # 明显没听清，直接返回空音频
            return {
                "user_text": user_text,
                "assistant_text": "",
                "audio_base64": "",
            }

        if session_id and session_id in session_context:
            prev_assistant = session_context[session_id]
            prev_norm = _normalize_text(prev_assistant)
            user_norm = _normalize_text(user_text_stripped)

            if prev_norm and user_norm:
                # 粗暴相似度：长度比 + 是否互相包含
                len_ratio = min(len(prev_norm), len(user_norm)) / max(
                    len(prev_norm), len(user_norm)
                )
                is_substring = user_norm in prev_norm or prev_norm in user_norm

                # 比如 0.7 + 子串关系，就认为是“上一轮 AI 的回声”
                if len_ratio > 0.7 and is_substring:
                    print(
                        "Detected echo of previous assistant reply, ignore this interruption."
                    )
                    return {
                        "user_text": user_text,
                        "assistant_text": "",
                        "audio_base64": "",
                    }

    # ---------- 2. LLM ----------
    system_content = (
        "You are a concise, helpful voice assistant. Answer in short sentences."
    )
    if is_interrupt:
        system_content += (
            " The user interrupted your previous response. "
            "Smoothly continue the conversation or switch topic based on the new request."
        )
        if session_id and session_id in session_context:
            system_content += (
                f" Previous response (interrupted): {session_context[session_id]}"
            )

    try:
        completion = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=256,
            temperature=0.7,
        )
        assistant_text = completion.choices[0].message.content
        print("assistant_text", assistant_text)

        if session_id:
            session_context[session_id] = assistant_text

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")

    # ---------- 3. TTS ----------
    try:
        tts_response = client.audio.speech.create(
            model=GROQ_TTS_MODEL,
            voice="Fritz-PlayAI",
            input=assistant_text,
            response_format="wav",
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
        tts_response.write_to_file(temp_path)
        with open(temp_path, "rb") as f:
            audio_wav_bytes = f.read()
        os.remove(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    audio_base64 = base64.b64encode(audio_wav_bytes).decode("utf-8")
    return {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "audio_base64": audio_base64,
    }


# /api/validate_interrupt 可以保留用于调试，但主流程已不依赖它
