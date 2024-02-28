
from UserImput import handle_userinput
import streamlit as st
from htmlTemplates import css
import logging


def main():
    logger = logging.getLogger("ChatBot")
    # Set the logging level to INFO
    logger.setLevel(logging.INFO)
    st.set_page_config(page_title="Chatbot",
                       page_icon="🤖",
                       layout="centered",
                       initial_sidebar_state="collapsed")
    st.write(css, unsafe_allow_html=True)

    # Initialize ST Session State
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "context" not in st.session_state:
        st.session_state.context = None

    st.title("💬 Chatbot Assistance")
    st.caption("🚀 Ask me anything about your documents")
    st.header("Q&A")

    if user_question := st.chat_input(placeholder="Can you give me a detail summary?"):
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
