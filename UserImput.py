from htmlTemplates import  bot_template, user_template
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import get_openai_callback
import streamlit as st
from dotenv import load_dotenv
import logging
import traceback

# Load env variables
load_dotenv()

logger = logging.getLogger('User Input')


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    # llm_hfai = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.9, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(question: str):
    logger.info("User Input: " + question)
    with get_openai_callback() as cb:
        try:
            # MakeSure there's a conversation chain before the first question
            response = st.session_state.conversation({'question': question})
            st.session_state.chat_history = response['chat_history']
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
            print(cb)
        except Exception as e:
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {e}")
            logger.error(f"Traceback: {traceback.print_exc()}")
