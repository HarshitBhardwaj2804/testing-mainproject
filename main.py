from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils import NamedJSONChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from datetime import datetime
import random, string
import json
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

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
chain = prompt | llm

def get_session_history(session_id: str):
    return NamedJSONChatMessageHistory(session_id=session_id, file_path="chat_sessions_test.json")

message_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

def generate_session_details():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    session_id = f"session_{timestamp}-{rand_suffix}"
    return session_id

app = Flask(__name__)

@app.route("/")
def index():
    session_id = generate_session_details()
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

@app.route("/load_chat/<session_id>")
def load_chat(session_id):
    return render_template("index.html", session_id=session_id)

@app.route("/get_previous_messages/<session_id>")
def get_previous_messages(session_id):
    file_path = "chat_sessions_test.json"
    if not os.path.exists(file_path):
        return jsonify([])

    with open(file_path, "r") as f:
        all_data = json.load(f)

    messages = all_data.get(session_id, [])
    return jsonify(messages)

@app.route("/history")
def history():
    file_path = "chat_sessions_test.json"
    if not os.path.exists(file_path):
        return render_template("history.html", sessions=[])

    with open(file_path, "r") as f:
        all_data = json.load(f)

    sessions = []
    for session_id, messages in all_data.items():
        if not messages or "chat_name" not in messages[0]:
            continue

        chat_name = messages[0]["chat_name"].strip()
        first_user_msg = next((m["content"] for m in messages if m.get("type") == "human"), "N/A")

        # Extract timestamp from session_id
        # Expecting: session_20250802114211310567-db0lt1
        parts = session_id.partition("_")[2].partition("-")[0]  # safer extraction
        datestamp_str = parts[:8]  # Only get date: YYYYMMDD
        timestamp_str = parts[8:12] # Only get the time: HHMM

        try:
            date_obj = datetime.strptime(datestamp_str, "%Y%m%d")
            formatted_date = date_obj.strftime("%B %d, %Y")
            time_obj = datetime.strptime(timestamp_str, "%H%M")
            formatted_time = time_obj.strftime("%I:%M %p")
        except Exception:
            formatted_date = "Unknown"

        date_time = f"{formatted_date} : {formatted_time}"
        sessions.append({
            "session_id": session_id,
            "chat_name": chat_name,
            "first_message": first_user_msg,
            "datetime" : date_time
        })

    # Newest first
    sessions.reverse()

    return render_template("history.html", sessions=sessions)

@app.route("/settings")
def settings():
    return render_template("settings.html")

if __name__ == "__main__":
    app.run(debug=True)
