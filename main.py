from fastapi import FastAPI
from pydantic import BaseModel
from ollama import Client  # Ollama Python SDK

app = FastAPI(title="Mental Health Chatbot API")
client = Client()

class PromptRequest(BaseModel):
    prompt: str
    model: str = "phi3"

@app.post("/chat")
def chat(request: PromptRequest):
    try:
        response = client.chat(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}]
        )
        text = response.get("message", {}).get("content", "").strip()
        if not text:
            return {"response": "No response from the model."}
        return {"response": text}
    except Exception as e:
        return {"error": str(e)}
