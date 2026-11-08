from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS so the frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    case_text: str
    reasoning: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        user_message = (
            f"Fictional case:\n{request.case_text}\n\n"
            f"Learner reasoning:\n{request.reasoning}"
        )
        response = client.chat.completions.create(
            model="gpt-5.6-terra",
            messages=[
                {"role": "system", "content": ("You are an educational clinical reasoning reviewer. The case is fictional and intended for learning, not patient care. Review the learner’s reasoning in English under four headings: 1. What the learner identified; 2. Points to reconsider; 3. Missing information; 4. A question for further reasoning. Distinguish facts in the case from hypotheses. Do not invent case details or attribute ideas to the learner that they did not express. State uncertainty and avoid presenting your response as a definitive diagnosis or treatment plan.")},
                {"role": "user", "content": user_message}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")
