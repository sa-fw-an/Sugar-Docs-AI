import streamlit as st
import requests
import os
import threading
import subprocess
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import logging
import socket

# Use Ollama Python library directly instead of subprocess
from ollama import chat, ResponseError, pull

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()


class ModalOllama:
    def generate(self, prompt: str, model_name: str) -> str:
        try:
            logger.info(f"Ensuring model '{model_name}' is available...")
            pull(model_name)
            logger.info(f"Model '{model_name}' is available. Generating response...")

            response = chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except ResponseError as e:
            logger.error(f"Ollama returned an error: {e}")
            return f"Error from Ollama: {e}"
        except Exception as e:
            logger.error(f"Runtime error calling Ollama: {e}")
            return f"Error: {e}"


model = ModalOllama()
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')


class SharedState:
    def __init__(self):
        self.parsed_data = {}
        self.corpus = []
        self.corpus_embeddings = None
        self.is_loaded = False


state = SharedState()


def load_data():
    logger.info("Loading data...")
    for root, _, files in os.walk('parsed_data'):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        state.parsed_data[file] = content
                        state.corpus.append(content)
    if state.corpus:
        state.corpus_embeddings = embedder.encode(state.corpus, convert_to_tensor=True)
        state.is_loaded = True
        logger.info("Documents loaded successfully.")
    else:
        logger.error("No documents found in corpus!")
        state.is_loaded = True


@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    if not state.is_loaded:
        return jsonify({'error': 'Document corpus not loaded'}), 500

    user_input = request.json.get('input', '')
    selected_model = request.json.get('model_name', 'deepseek-r1:1.5b')
    try:
        query_embedding = embedder.encode(user_input, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, state.corpus_embeddings, top_k=5)

        sources = []
        relevant_info = ""
        for hit in hits[0]:
            if hit['score'] > 0.2:
                src_name = list(state.parsed_data.keys())[hit['corpus_id']]
                sources.append(src_name)
                doc_text = state.corpus[hit['corpus_id']][:500]
                relevant_info += f"\n[Source: {src_name}]\n{doc_text}\n"

        
        prompt = f"""You are a Sugar Labs assistant. Answer the user's question based on the following documentation sections.
        If multiple sources provide relevant information, combine them in your response.
        Remember Sugar uses AGPLv3 license, so all contributions must be compatible with this license.
        Documentation sections:
        {relevant_info}
        Question: {user_input}
        Provide a clear and specific answer, citing the relevant documentation where possible."""
        response_text = model.generate(prompt, selected_model)
        return jsonify({'response': response_text, 'sources': sources})

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def run_flask(port):
    try:
        app.run(host='0.0.0.0', port=port, use_reloader=False)
    except Exception as e:
        logger.error(f"Failed to start Flask server on port {port}: {e}")


def start_flask_server():
    if 'flask_started' not in st.session_state:
        st.session_state.flask_started = False

    if not st.session_state.flask_started:
        default_port = 5000
        port = default_port
        while is_port_in_use(port):
            port += 1
            if port > 6000:
                st.error("No available ports found for Flask server.")
                return

        st.session_state.flask_port = port
        flask_thread = threading.Thread(target=run_flask, args=(port,))
        flask_thread.daemon = True
        flask_thread.start()
        st.session_state.flask_started = True
        logger.info(f"Flask server started on port {port}")


if 'data_loaded' not in st.session_state:
    load_data()
    st.session_state.data_loaded = True

start_flask_server()

st.title(" Local App Sugar Docs")

def check_ollama_connection():
    try:
        pull("deepseek-r1:1.5b")
        return True
    except Exception:
        return False


def start_ollama_serve():
    try:
        # Start 'ollama serve' in a new subprocess
        ollama_cmd = ['ollama', 'serve']
        process = subprocess.Popen(ollama_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Started Ollama serve.")
        # Give some time for Ollama to start
        time.sleep(10)  # Increased sleep time to allow Ollama to initialize properly
        return process
    except Exception as e:
        logger.error(f"Failed to start Ollama serve: {e}")
        return None


# Session state to keep track of Ollama process
if 'ollama_process' not in st.session_state:
    st.session_state.ollama_process = None

# Function to terminate Ollama process on app close
def terminate_ollama():
    if st.session_state.ollama_process:
        st.session_state.ollama_process.terminate()
        st.session_state.ollama_process = None
        logger.info("Terminated Ollama serve.")


if not check_ollama_connection():
    st.warning("Ollama is not running.")
    if st.button("Start Ollama & Pull Model"):
        with st.spinner("Starting Ollama..."):
            if st.session_state.ollama_process is None:
                ollama_proc = start_ollama_serve()
                if ollama_proc:
                    st.session_state.ollama_process = ollama_proc
                    st.success("Ollama started successfully.")
                else:
                    st.error("Failed to start Ollama.")
            else:
                st.warning("Ollama is already running.")

        if st.session_state.ollama_process:
            model_choice = st.session_state.get('model_choice', 'deepseek-r1:1.5b')
            with st.spinner(f"Pulling model '{model_choice}'..."):
                try:
                    pull(model_choice)
                    st.success(f"Model '{model_choice}' pulled successfully.")
                except ResponseError as e:
                    st.error(f"Error pulling model: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
else:
    st.success("Connected to Ollama successfully.")

model_choice = st.selectbox(
    "Select model",
    # add some low sized LLMs
    ["deepseek-r1:7b", "deepseek-r1:1.5b", "deepseek-r1:8b", "deepseek-r1:70b" , "mistral:7b"]
)

# Store selected model in session state
st.session_state['model_choice'] = model_choice

# Function to check if a specific model is pulled
def is_model_pulled(model_name):
    try:
        # Attempt to pull the model; if it's already pulled, Ollama will skip
        pull(model_name)
        return True
    except ResponseError:
        return False
    except Exception:
        return False


model_status = is_model_pulled(model_choice)
if model_status:
    st.write(f"**Model Status:** '{model_choice}' is available.")
else:
    st.write(f"**Model Status:** '{model_choice}' is not available. Click the button above to pull the model.")

user_input = st.text_area("Type your question:")
submit_button = st.button("Submit")

if submit_button:
    if user_input.strip():
        if not check_ollama_connection():
            st.error("Ollama is not running. Please start Ollama and ensure the model is pulled.")
        elif not model_status:
            st.error(f"Model '{model_choice}' is not available. Please pull the model before submitting.")
        else:
            with st.spinner("Processing your request..."):
                payload = {
                    "input": user_input,
                    "model_name": model_choice
                }
                # The Flask API is running on the determined port
                flask_port = st.session_state.get('flask_port', 5000)
                flask_url = f"http://localhost:{flask_port}/api/chatbot"

                try:
                    r = requests.post(flask_url, json=payload)
                    if r.status_code == 200:
                        response_json = r.json()
                        st.write("**Response:**")
                        st.write(response_json.get('response'))

                        if response_json.get('sources'):
                            st.write("**Sources Used:**")
                            for s in response_json['sources']:
                                st.write(f"- {s}")
                    else:
                        st.error(f"API error: {r.status_code} - {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API (ensure Flask is running and Ollama is installed/pulled).")
    else:
        st.warning("Enter a question first.")
