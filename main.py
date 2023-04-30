import json
import quart
import quart_cors
from quart import request
from datetime import datetime

app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")

# Keep track of conversation history. Does not persist if Python session is restarted.
_HISTORY = {}

@app.post("/memory/<string:username>")
async def add_memory(username):
    request_data = await quart.request.get_json(force=True)
    user_input = request_data.get("user_input")
    chatgpt_response = request_data.get("chatgpt_response")
    timestamp = datetime.now().isoformat()

    if username not in _HISTORY:
        _HISTORY[username] = []

    interaction = {
        "user_input": user_input,
        "chatgpt_response": chatgpt_response,
        "timestamp": timestamp
    }
    _HISTORY[username].append(interaction)
    return quart.Response(response='OK', status=200)

@app.get("/memory/<string:username>")
async def get_memory(username):
    return quart.Response(response=json.dumps(_HISTORY.get(username, [])), status=200)

@app.delete("/memory/<string:username>")
async def delete_memory(username):
    request_data = await quart.request.get_json(force=True)
    interaction_idx = request_data.get("interaction_idx")

    # Fail silently, it's a simple plugin
    if 0 <= interaction_idx < len(_HISTORY[username]):
        _HISTORY[username].pop(interaction_idx)
    return quart.Response(response='OK', status=200)

@app.get("/logo.png")
async def plugin_logo():
    filename = 'logo.png'
    return await quart.send_file(filename, mimetype='image/png')

@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    host = request.headers['Host']
    with open("./.well-known/ai-plugin.json") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/json")

@app.get("/openapi.yaml")
async def openapi_spec():
    host = request.headers['Host']
    with open("openapi.yaml") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/yaml")

def main():
    app.run(debug=True, host="0.0.0.0", port=5004)

if __name__ == "__main__":
    main()
