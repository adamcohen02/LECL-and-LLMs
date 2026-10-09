from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from config import OLLAMA_BASE_URL, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID
from collections import OrderedDict
from hashlib import sha256
from json import dumps
from threading import RLock

# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")
    action: str = Field(description="Recommended action for the support rep")
# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)

# Function to initialize a model
def initialize_model(model_id):
    return ChatOllama(
        model=model_id,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        format="json",
    )

# Initialize models
llama_llm = initialize_model(LLAMA_MODEL_ID)
granite_llm = initialize_model(GRANITE_MODEL_ID)
mistral_llm = initialize_model(MISTRAL_MODEL_ID)

_conversation_histories = OrderedDict()
_history_lock = RLock()
_max_conversations = 256
_max_history_messages = 20


def get_session_history(session_id):
    """Return the in-memory LangChain history for one browser conversation."""
    with _history_lock:
        history = _conversation_histories.get(session_id)
        if history is None:
            history = InMemoryChatMessageHistory()
            _conversation_histories[session_id] = history
        if len(history.messages) > _max_history_messages:
            del history.messages[:-_max_history_messages]
        _conversation_histories.move_to_end(session_id)
        while len(_conversation_histories) > _max_conversations:
            _conversation_histories.popitem(last=False)
        return history


def clear_session_history(session_id):
    with _history_lock:
        _conversation_histories.pop(session_id, None)


def get_history_fingerprint(session_id):
    history = get_session_history(session_id)
    messages = [(message.type, message.content) for message in history.messages]
    serialized = dumps(messages, sort_keys=True, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def get_ai_response(model, system_prompt, user_prompt, session_id):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}\n{format_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_prompt}"),
    ])
    chain = prompt | model
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="user_prompt",
        history_messages_key="history",
    )
    response = chain_with_history.invoke(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "format_prompt": json_parser.get_format_instructions(),
        },
        config={"configurable": {"session_id": session_id}},
    )
    return json_parser.parse(response.content)

# Model-specific response functions
def llama_response(system_prompt, user_prompt, session_id):
    return get_ai_response(llama_llm, system_prompt, user_prompt, session_id)

def granite_response(system_prompt, user_prompt, session_id):
    return get_ai_response(granite_llm, system_prompt, user_prompt, session_id)

def mistral_response(system_prompt, user_prompt, session_id):
    return get_ai_response(mistral_llm, system_prompt, user_prompt, session_id)
