from fastapi import FastAPI
from pydantic import BaseModel
from ollama import ChatCompletion  # Ollama Python SDK

app = FastAPI(title="Mental Health Chatbot API")

class PromptRequest(BaseModel):
    prompt: str
    model: str = "phi3"

@app.post("/chat")
def chat(request: PromptRequest):
    try:
        completion = ChatCompletion.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}]
        )
        text = completion.choices[0].message.get("content", "").strip()
        if not text:
            return {"response": "No response from the model."}
        return {"response": text}
    except Exception as e:
        return {"error": str(e)}
