from langchain_ollama import ChatOllama
import streamlit as st

# Initialize with explicit settings
llm = ChatOllama(
    model="mistral",
    base_url="http://127.0.0.1:11434",  # Critical for Windows
    timeout=300,  # Prevents timeouts
    verbose=True  # Logs connection attempts
)

# Test connection
try:
    response = llm.invoke("Hello")
    st.success("Ollama connection successful!")
    st.write(response)
except Exception as e:
    st.error(f"Ollama failed: {str(e)}")