## General functions
import json
import os
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

class JSONChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, file_path: str = "../chat_sessions.json"):
        self.session_id = session_id
        self.file_path = file_path
        self.messages = self._load_messages()

    def _load_messages(self) -> List[BaseMessage]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r") as f:
            all_data = json.load(f)
        session_data = all_data.get(self.session_id, [])
        return [self._dict_to_message(m) for m in session_data]

    def _save_messages(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[self.session_id] = [self._message_to_dict(m) for m in self.messages]

        with open(self.file_path, "w") as f:
            json.dump(all_data, f, indent=2)

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self._save_messages()

    def clear(self) -> None:
        self.messages = []
        self._save_messages()

    def _message_to_dict(self, message: BaseMessage):
        return {"type": message.type, "content": message.content}

    def _dict_to_message(self, data: dict) -> BaseMessage:
        if data["type"] == "human":
            return HumanMessage(content=data["content"])
        elif data["type"] == "ai":
            return AIMessage(content=data["content"])
        else:
            raise ValueError("Unknown message type")
