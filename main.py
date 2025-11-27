import base64
import os
import tempfile
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

# 开发阶段直接全开放 CORS，方便你用真机调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用环境变量里的 GROQ_API_KEY，Railway 上配置
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@app.post("/api/voice_chat")
async def voice_chat(
    file: UploadFile = File(...),
    is_interruption: str = Form(default="false")
) -> Dict[str, str]:
    """
    接收一个音频文件（wav/m4a 等），
    1) STT: whisper-large-v3
    2) LLM: llama3-8b-8192
    3) TTS: playai-tts
    返回:
    {
      "user_text": "...",
      "assistant_text": "...",
      "audio_base64": "..."
    }
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    filename = file.filename or "audio.m4a"
    # ---------- 1. STT：whisper-large-v3 ----------
    # 参考官方文档，通过 Groq SDK 调用 audio.transcriptions.create
    try:
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes), # (file_name, bytes)
            model="whisper-large-v3", # 固定为你要求的模型
            response_format="json",
            temperature=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    user_text = getattr(transcription, "text", None)
    print("user_text", user_text)
    if not user_text:
        raise HTTPException(status_code=500, detail="STT result has no text")
    # ---------- 2. LLM：llama3-8b-8192 ----------
    system_content = "You are a concise, helpful voice assistant. Answer in short sentences."
    if is_interruption.lower() == "true":
        system_content += " The user interrupted your previous response. Start your response with a smooth transition phrase like 'Okay, let me check that for you.' or 'Sure, switching topics now.'"
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=256,
            temperature=0.7,
        )
        assistant_text = completion.choices[0].message.content
        print("assistant_text", assistant_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")
    # ---------- 3. TTS：playai-tts ----------
    # 官方示例：client.audio.speech.create(...).write_to_file("speech.wav")
    try:
        tts_response = client.audio.speech.create(
            model="playai-tts", # 或 "playai-tts-arabic"
            voice="Fritz-PlayAI", # 任意一个官方支持的 English voice
            input=assistant_text,
            response_format="wav", # 默认也是 wav，这里显式写上
        )
        # 为了简单，按官方方式写入一个临时文件再读回来
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