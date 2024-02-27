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

    st.header("Knowledge Base :open_book:")
    st.subheader("From Documents text and PDF only :books:")

    files_uploaded = st.file_uploader(
        "Upload your Files here and click on 'Process'",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="visible")

    if st.button("Process"):
        with st.spinner("Loading..."):
            # Extract text and metadata from PDF
            raw_text, pdf_metadata = get_pdf_text_metadata(pdf_docs=files_uploaded)
            # Get Chunks
            chunks, ids = get_text_chunks_ids(raw_text)

            # create vector store obj
            vs = VectorStore()
            # embeddings, ids, metadata = vs.get_embeddings_and_ids(text_chunks=chunks, metadata=metadata)
            vs.create_pinecone_index()
            # vs.pinecone_upsert(embeddings, ids, metadata)

            vectorstore = vs.get_vectorstore_pinecone(text=chunks, ids=ids)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                vectorstore)
        st.success("Done!")


if __name__ == '__main__':
    main()
