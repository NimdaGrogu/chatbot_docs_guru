from VectordbOps import VectorStore
from CorpusTextOps import get_text_chunks_ids, get_pdf_text_metadata, identify_file_type
from UserImput import get_conversation_chain
import streamlit as st
from htmlTemplates import css
import logging


def main():
    logger = logging.getLogger("ChatBot")
    # Set the logging level to INFO
    logger.setLevel(logging.INFO)
    st.set_page_config(page_title="Upload Files", page_icon="book",
                       layout="wide"
                       )
    st.write(css, unsafe_allow_html=True)

    st.header("Knowledge Base :open_book:")

    st.subheader("From Documents text and PDF only :books:")

    docs = st.file_uploader(
        "Upload your Files here and click on 'Process'",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="visible")

    if st.button("Process"):
        with st.spinner("Loading..."):

            # get pdf text
            raw_text, metadata = get_pdf_text_metadata(docs)

            # get the text chunks
            text_chunks, ids = get_text_chunks_ids(raw_text)

            # create vector store
            vs = VectorStore()
            # Create an Index db using text_chunks and save it locally for further use
            vectorstore = vs.build_knowledgebase_faiss(text_chunks=text_chunks, save=True)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                vectorstore)
        st.success("Done!")


if __name__ == '__main__':
    main()
