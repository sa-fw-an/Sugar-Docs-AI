from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

# Load the sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load parsed data and create embeddings
parsed_data = {}
corpus = []
corpus_embeddings = []

for root, _, files in os.walk('parsed_data'):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                parsed_data[file] = content
                corpus.append(content)

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    response = model.generate_content(user_input)
    if not response:
        return jsonify({'error': 'Failed to get response from Gemini API'}), 500

    query_embedding = embedder.encode(user_input, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
    
    threshold = 0.75
    relevant_info = ""
    for hit in hits[0]:
        if hit['score'] > threshold:
            relevant_info += f"\n\nAdditional info from {list(parsed_data.keys())[hit['corpus_id']]}:\n{corpus[hit['corpus_id']][:500]}..."

    enhanced_response = response.text
    if relevant_info:
        enhanced_response += relevant_info

    return jsonify({'response': enhanced_response})

@app.route('/areyoualive', methods=['GET'])
def ping():
    return "Pong!", 200

if __name__ == '__main__':
    app.run(debug=True)