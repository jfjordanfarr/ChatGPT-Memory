import openai  # Import the OpenAI library
import numpy as np
import logging
import time

# Define the predefined set of categories
VALID_CATEGORIES = {'Personal', 'Work', 'Education', 'Entertainment', 'Other'}

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
