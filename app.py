from flask import Flask, request, jsonify, render_template, session
from model import (
    llama_response,
    granite_response,
    mistral_response,
    get_history_fingerprint,
    get_session_history,
    clear_session_history,
)
from collections import OrderedDict
from json import dumps
import os
import secrets
from threading import RLock
import time

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


class TTLResponseCache:
    def __init__(self, max_entries=256, ttl_seconds=300):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries = OrderedDict()
        self._lock = RLock()

    def get(self, key):
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                return None

            self._entries.move_to_end(key)
            return value.copy()

    def set(self, key, value):
        now = time.monotonic()
        with self._lock:
            expired_keys = [
                cache_key
                for cache_key, (expires_at, _) in self._entries.items()
                if expires_at <= now
            ]
            for cache_key in expired_keys:
                del self._entries[cache_key]

            self._entries[key] = (now + self.ttl_seconds, value.copy())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear_conversation(self, conversation_id):
        with self._lock:
            keys = [key for key in self._entries if key[0] == conversation_id]
            for key in keys:
                del self._entries[key]


response_cache = TTLResponseCache()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/reset', methods=['POST'])
def reset_conversation():
    conversation_id = session.pop("conversation_id", None)
    if conversation_id is not None:
        clear_session_history(conversation_id)
        response_cache.clear_conversation(conversation_id)
    return jsonify({"ok": True})


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    user_message = data.get('message')
    model = data.get('model')
    
    if not isinstance(user_message, str) or not user_message.strip() or not isinstance(model, str) or not model:
        return jsonify({"error": "Missing message or model selection"}), 400

    if model not in ('llama', 'granite', 'mistral'):
        return jsonify({"error": "Invalid model selection"}), 400
    
    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response."
    conversation_id = session.get("conversation_id")
    if conversation_id is None:
        conversation_id = secrets.token_urlsafe(32)
        session["conversation_id"] = conversation_id

    cache_key = (
        conversation_id,
        model,
        system_prompt,
        get_history_fingerprint(conversation_id),
        user_message,
    )
    
    start_time = time.time()
    
    try:
        result = response_cache.get(cache_key)
        cache_hit = result is not None

        if not cache_hit:
            if model == 'llama':
                result = llama_response(system_prompt, user_message, conversation_id)
            elif model == 'granite':
                result = granite_response(system_prompt, user_message, conversation_id)
            else:
                result = mistral_response(system_prompt, user_message, conversation_id)
            response_cache.set(cache_key, result)
        else:
            history = get_session_history(conversation_id)
            history.add_user_message(user_message)
            history.add_ai_message(dumps(result))
        
        result['duration'] = time.time() - start_time
        result['cached'] = cache_hit
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
