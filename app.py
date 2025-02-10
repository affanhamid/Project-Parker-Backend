import os
import uuid

from chat_caller import query_gpt_chat
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# Update allowed origins to include both your local and Vercel domains
origins = ["http://localhost:3000", "https://ai-assistant-orcin-gamma.vercel.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only the specified origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest, request: Request):
    conversation_id = str(uuid.uuid4())
    admin_token = request.query_params.get("admin_token", None)
    _, answer = query_gpt_chat(
        chat_req.message, [], False, conversation_id, admin_token
    )
    return {"answer": answer, "conversation_id": conversation_id}
