import streamlit as st
import requests
import os
import threading
from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

parsed_data = {}
corpus = []
corpus_embeddings = []

def load_data():
    global parsed_data, corpus, corpus_embeddings
    parsed_data = {}
    corpus = []
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

def run_flask():
    load_data()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)

threading.Thread(target=run_flask).start()

st.title("Sugar Labs Chatbot")

st.write("Ask a question about contributing to Sugar Labs:")

user_input = st.text_area("Your question:")

if st.button("Submit"):
    if user_input.strip():
        api_url = os.getenv("API_URL", "http://localhost:5000/api/chatbot")
        try:
            response = requests.post(api_url, json={"input": user_input})
            if response.status_code == 200:
                chatbot_response = response.json().get("response", "No response from chatbot.")
                st.write("Chatbot response:")
                st.write(chatbot_response)
            else:
                st.write("Error:", response.status_code)
        except requests.exceptions.ConnectionError:
            st.write("Error: Unable to connect to the API.")
    else:
        st.write("Please enter a question.")