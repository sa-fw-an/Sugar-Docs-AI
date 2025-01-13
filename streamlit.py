import streamlit as st
import requests
import os

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