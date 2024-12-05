import streamlit as st
from openai import OpenAI
from pinecone_handler import generate_prompt
from local_llm import ask_local_llm


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


def display_messages():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


def handle_user_input(prompt):
    try:
        with st.spinner("Generating response..."):
            st.chat_message("user").write(prompt)
            # Generate context-aware prompt
            prompt_with_context = generate_prompt(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt_with_context})

            response = ask_local_llm(prompt_with_context)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Main Streamlit app
def start():
    st.title("MSE 💬Chatbot")

    initialize_session_state()

    # Display chat history
    display_messages()

    if user_input := st.chat_input("Type your message here..."):
        handle_user_input(user_input)


start()