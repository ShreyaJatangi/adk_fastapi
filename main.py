from google.adk.agents import LlmAgent
from google.adk.runners import Runner  
from google.adk.sessions import  RedisSessionService
from google.genai import types
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import traceback
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import traceback

from dotenv import load_dotenv
load_dotenv()

APP_NAME = "Brandon bot"

def get_result(num: int) -> int:
    """Take the input from user and return square of that number."""
    print(f"Tool 'get_result' called with number: {num}")
    return num * num

root_agent = LlmAgent(
    name="Square_num",
    model="gemini-2.0-flash",
    description="Doing mathematical calculation for the users request",
    instruction="""You are a helpful Maths Agent. Your primary function is to use your tools to perform calculations and then report the results to the user.
    
    Follow these steps STRICTLY:
    1. Analyze the user's request to identify the required calculation.
    2. Call the 'get_result' tool with the correct number.
    3. After the tool returns a result, you MUST formulate a complete sentence that clearly states the answer.
    
    Your final response to the user must ALWAYS be a user-facing sentence. NEVER end your turn immediately after a tool call without reporting the result. For example, if the tool returns 81, you MUST say 'The result of the calculation is 81.' or something similar.""",
    tools=[get_result],
)
redis_url = os.environ.get("REDIS_URL")
session_service_stateful = RedisSessionService(redis_url=redis_url)
runner = Runner(agent=root_agent, session_service=session_service_stateful, app_name=APP_NAME)

app = FastAPI()

class CreateSessionRequest(BaseModel):
    user_id: str

class SessionResponse(BaseModel):
    user_id: str
    session_id: str

class ChatRequest(BaseModel):
    prompt: str
    user_id: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str

@app.post("/sessions", response_model=SessionResponse, tags=["Session Management"])
async def create_new_session(request: CreateSessionRequest):
    user_id = request.user_id
    session_id = str(uuid.uuid4())
    await session_service_stateful.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    print(f"API created new session {session_id} for user {user_id}")
    return SessionResponse(user_id=user_id, session_id=session_id)

@app.post("/chat", response_model=ChatResponse, tags=["Agent Interaction"])
async def chat_with_agent(request: ChatRequest):
    user_id = request.user_id
    session_id = request.session_id
    try:
        if not await session_service_stateful.get_session(app_name=APP_NAME,user_id=user_id, session_id=session_id):
            raise HTTPException(status_code=404, detail=f"Session with id '{session_id}' not found.")

        print(f"Using existing session {session_id} for user {user_id}")
        new_message = types.Content(role="user", parts=[types.Part(text=request.prompt)])
        final_response = ""

        for event in runner.run(
            user_id=user_id, session_id=session_id, new_message=new_message
        ):
            # --- FINAL CORRECT LOGIC BASED ON YOUR LOGS ---
            # Check if the event has a content object with parts
            if hasattr(event, 'content') and event.content and event.content.parts:
                # Check if the first part has text content
                part = event.content.parts[0]
                if hasattr(part, 'text') and part.text:
                    # We found the final text response from the model!
                    final_response = part.text.strip()
                    print(f"  -> Captured final model response: '{final_response}'")

        if not final_response:
             final_response = "The action was completed, but the agent provided no verbal response."

        return ChatResponse(
            response=final_response, user_id=user_id, session_id=session_id
        )
    except Exception as e:
        print(f"An error occurred during chat: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
