from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils import JSONChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from datetime import datetime
import random, string
load_dotenv()

## Setting-Up Langchain-tracing.
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CARL"

groq_api_key = os.getenv("GROQ_API_KEY_MAIN_PROJECT")

prompt = ChatPromptTemplate(
    [
        ("system", ("You are CARL, an excellent assistant. Your task is to help your master as best you can in his/her questions.")),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}")
    ]
)

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
chain = prompt | llm

def get_session_history(session_id: str):
    return JSONChatMessageHistory(session_id=session_id, file_path="chat_sessions.json")

message_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

def generate_session_details():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    session_id = f"session_{timestamp}-{rand_suffix}"
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    return [session_id, date, time]

app = Flask(__name__)

@app.route("/")
def index():
    session_id, date, time = generate_session_details()
    return render_template("index.html", session_id = session_id)

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.json.get("message")
    session_id = request.json.get("session_id")
    
    response = message_chain.invoke(
    {"question": user_message},
    config={"configurable": {"session_id": session_id}}
    )
    
    return jsonify({"reply": response.content})

@app.route("/extensions")
def extensions():
    return render_template("extensions.html")

@app.route("/history")
def history():
    return render_template("history.html")

if __name__ == "__main__":
    app.run(debug=True)
