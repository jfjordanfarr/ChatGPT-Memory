import json
import quart
import quart_cors
from quart import request
from datetime import datetime
from tinydb import TinyDB, Query
import logging
import openai  # Import the OpenAI library
import os
import time
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")

# Use TinyDB for persistence.
db = TinyDB('memory_db.json')

# Set up logging
logging.basicConfig(filename='memory_plugin.log', level=logging.DEBUG)

# Function to generate embeddings using OpenAI's API
def get_ada_embedding(text):
    text = text.replace("\n", " ")
    start_time = time.time()  # Measure the start time
    embedding = openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]
    end_time = time.time()  # Measure the end time
    logging.debug(f"get_ada_embedding duration: {end_time - start_time} seconds")  # Log the duration
    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Increase the maximum number of interactions stored per user
max_interactions = 500  # Increase as needed

# Define the predefined set of categories
VALID_CATEGORIES = {'Personal', 'Work', 'Education', 'Entertainment', 'Other'}

@app.post("/memory/<string:username>")
async def add_memory(username):
    start_time = time.time()  # Measure the start time
    request_data = await quart.request.get_json(force=True)
    user_input = request_data.get("user_input")
    chatgpt_response = request_data.get("chatgpt_response")
    document_name = request_data.get("document_name", "conversation")  # Default to "conversation"
    author = request_data.get("author", "ChatGPT")  # Default to "ChatGPT"
    timestamp = datetime.now().isoformat()
    # Retrieve the category field from the request
    category = request_data.get("category")
    
    # Validate the category field
    if category not in VALID_CATEGORIES:
        return quart.Response(response='Invalid input: category must be one of the predefined categories', status=400)
    # Check if user_input or chatgpt_response are missing
    if user_input is None or chatgpt_response is None:
        return quart.Response(response='Invalid input: user_input and chatgpt_response must be provided', status=400)
    # Check that both user_input and chatgpt_response are non-empty strings
    if not user_input or not chatgpt_response:
        return quart.Response(response='Invalid input: user_input and chatgpt_response must be non-empty strings', status=400)

    # Generate embeddings for user input and ChatGPT response
    user_input_embedding = get_ada_embedding(user_input)
    chatgpt_response_embedding = get_ada_embedding(chatgpt_response)
    # Retrieve or create user's memory.
    User = Query()
    user_memory = db.search(User.username == username)
    if not user_memory:
        db.insert({'username': username, 'interactions': []})
        user_memory = db.search(User.username == username)

    # Limit the number of interactions stored per user
    interactions = user_memory[0]['interactions']
    if len(interactions) >= max_interactions:
        interactions.pop(0)

    # Check for duplicate interactions
    existing_interaction = next((interaction for interaction in interactions if interaction["user_input"] == user_input and interaction["chatgpt_response"] == chatgpt_response), None)
    if existing_interaction:
        return quart.Response(response='Duplicate interaction', status=400)

    interaction = {
        "user_input": user_input,
        "user_input_embedding": user_input_embedding,
        "chatgpt_response": chatgpt_response,
        "chatgpt_response_embedding": chatgpt_response_embedding,
        "timestamp": timestamp,
        "document_name": document_name,
        "author": author,
        "category": category, # Store the category
        }
    interactions.append(interaction)
    db.update({'interactions': interactions}, User.username == username)

    # Log the memory operation without logging the interaction details
    logging.info(f'Added interaction to memory for user {username}')
    logging.debug(f'Current memory state for user {username}: {len(interactions)} interactions')
    end_time = time.time()  # Measure the end time
    logging.debug(f"add_memory duration: {end_time - start_time} seconds")  # Log the duration
    return quart.Response(response='OK', status=200)

async def get_memory(username):
    query_text = request.args.get('query_text', type=str)
    query_embedding = get_ada_embedding(query_text) if query_text else None
    query_category = request.args.get('category', type=str) # Optional query category
    User = Query()
    user_memory = db.search(User.username == username)
    interactions = user_memory[0]['interactions'] if user_memory else []

    if query_embedding:
        # Filter interactions based on cosine similarity with query_embedding
        threshold = 0.5  # Adjust the threshold as needed
        relevant_interactions = [interaction for interaction in interactions if cosine_similarity(np.array(interaction["user_input_embedding"]), query_embedding) >= threshold]
    else:
        relevant_interactions = interactions

    # Filter interactions based on category if provided
    if query_category:
        relevant_interactions = [interaction for interaction in relevant_interactions if interaction["category"] == query_category]

    # Sort interactions by document_name and timestamp
    relevant_interactions.sort(key=lambda x: (x['document_name'], x['timestamp']))

    # Stitch together interactions into a single document
    stitched_document = ""
    for interaction in relevant_interactions:
        stitched_document += interaction["user_input"] + "\n" + interaction["chatgpt_response"] + "\n"

    # Log the memory retrieval without logging the interaction details
    logging.info(f'Retrieved memory for user {username}')
    logging.debug(f'Current memory state for user {username}: {len(relevant_interactions)} interactions')

    # Optionally, use the get_summary function to return a summarized version of the document
    # stitched_document = get_summary(stitched_document)

    return quart.Response(response=stitched_document, status=200)


@app.post("/memory/update/<string:username>")
async def update_memory(username):
    request_data = await quart.request.get_json(force=True)
    document_name = request_data.get("document_name")
    interactions_to_update = request_data.get("interactions", [])

    # Check if document_name is missing
    if document_name is None:
        return quart.Response(response='Invalid input: document_name must be provided', status=400)

    # Delete all previous chunks of the document
    User = Query()
    user_memory = db.search(User.username == username)
    if user_memory:
        interactions = user_memory[0]['interactions']
        interactions = [interaction for interaction in interactions if interaction['document_name'] != document_name]
        db.update({'interactions': interactions}, User.username == username)

    for interaction in interactions_to_update:
        add_memory(username, interaction)

    return quart.Response(response='OK', status=200)

@app.delete("/memory/<string:username>")
async def delete_memory(username):
    request_data = await quart.request.get_json(force=True)
    interaction_idx = request_data.get("interaction_idx")

    # Check if interaction_idx is missing or invalid
    if interaction_idx is None or not isinstance(interaction_idx, int):
        return quart.Response(response='Invalid input: interaction_idx must be provided as an integer', status=400)


    User = Query()
    user_memory = db.search(User.username == username)
    if user_memory:
        interactions = user_memory[0]['interactions']
        if 0 <= interaction_idx < len(interactions):
            interaction = interactions.pop(interaction_idx)
            db.update({'interactions': interactions}, User.username == username)
            
            # Log the memory deletion
            logging.info(f'Deleted interaction from memory for user {username}: {interaction}')
            logging.debug(f'Current memory state for user {username}: {interactions}')
    
    return quart.Response(response='OK', status=200)

# Function to generate summaries using OpenAI's GPT-3.5-turbo model
def get_summary(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the following text: {text}"},
        ],
    )
    return response['choices'][0]['message']['content']

@app.get("/memory/summarize_user/<string:username>")
async def summarize_user(username):
    User = Query()
    user_memory = db.search(User.username == username)
    interactions = user_memory[0]['interactions'] if user_memory else []
    
    # Extract user inputs from interactions
    user_inputs = [interaction["user_input"] for interaction in interactions]
    user_text = "\n".join(user_inputs)
    
    # Generate summary of user interactions
    summary = get_summary(user_text)
    
    return quart.Response(response=summary, status=200)

@app.get("/memory/summarize_chatgpt/<string:username>")
async def summarize_chatgpt(username):
    User = Query()
    user_memory = db.search(User.username == username)
    interactions = user_memory[0]['interactions'] if user_memory else []
    
    # Extract ChatGPT responses from interactions
    chatgpt_responses = [interaction["chatgpt_response"] for interaction in interactions]
    chatgpt_text = "\n".join(chatgpt_responses)
    
    # Generate summary of ChatGPT responses
    summary = get_summary(chatgpt_text)
    
    return quart.Response(response=summary, status=200)

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
