# Sugar Labs Chatbot

This project is a chatbot for Sugar Labs using the Gemini API. The chatbot interacts with users who are willing to contribute to Sugar Labs by providing information from the Sugar Labs documentation.

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn
- Virtual environment (optional but recommended)

## Setup

### Backend

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/sugar-labs-chatbot.git
    cd sugar-labs-chatbot
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required Python packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a [.env](http://_vscodecontentref_/0) file in the root directory and add your Gemini API key:

    ```env
    GEMINI_API_KEY=your_gemini_api_key
    ```

5. **Run the Flask server:**

    ```sh
    python server.py
    ```

    The server will start on `http://localhost:5000`.

### Frontend

1. **Navigate to the frontend directory:**

    ```sh
    cd sugar-docs-frontend-ai
    ```

2. **Install the required Node.js packages:**

    ```sh
    npm install
    ```

3. **Set up environment variables:**

    Create a [.env.local](http://_vscodecontentref_/2) file in the [sugar-docs-frontend-ai](http://_vscodecontentref_/3) directory and add the backend URL:

    ```env
    NEXT_PUBLIC_API_URL=http://localhost:5000/api/chatbot
    ```

4. **Run the Next.js development server:**

    ```sh
    npm run dev
    ```

    The frontend app will start on `http://localhost:3000`.

## Usage

### Using Backend and Frontend
1. **Start the backend server:**

    ```sh
    python server.py
    ```

2. **Start the frontend app:**

    ```sh
    cd sugar-docs-frontend-ai
    npm run dev
    ```

3. **Open your browser and navigate to `http://localhost:3000` to interact with the chatbot.**

### Running the Streamlit App

1. **Run the Streamlit app:**

    ```sh
    streamlit run streamlit.py
    ```

2. **Open your browser and navigate to the URL provided by Streamlit (e.g., `http://localhost:8501`) to interact with the chatbot.**


### Using the purely local Streamlit App

1. **Run the Streamlit app:**

    ```sh
    streamlit run containedlocal.py
    ```

2. **Open your browser and navigate to the URL provided by Streamlit (e.g., `http://localhost:8501`) to interact with the chatbot.**


#### NOTES: make sure you have ollama installed and do pip isntall requirements_for_local for this one.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License.