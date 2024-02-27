from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import Pinecone
from PineconeCore import PineconeOperations
from dotenv import load_dotenv, find_dotenv
import traceback
import os
import logging
import time

# Basic configuration of logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load env variables
_ = load_dotenv(find_dotenv())  # read local .env file


class VectorStore(PineconeOperations):
    """
    This class is designed to store vector representations of text.
    It utilizes embeddings from both OpenAI and HuggingFace models.
    """

    def __init__(self, **kwargs):
        """
        Constructor for the VectorStore class.
        Initializes embeddings from OpenAI and HuggingFace models.
        """

        # Create a logger for this method
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"VectorStore: Initiation {self.__class__.__name__}")
        #  Instance of the Parent class and get ready methods and attributes
        super().__init__()
        # Initialize OpenAI embeddings
        self.embedding_model = None
        self.embeddings_openai = OpenAIEmbeddings(model='text-embedding-ada-002', show_progress_bar=True)
        # Initialize HuggingFace Instruct embeddings with a specified model

        self.embeddings_huggingface = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

    # %%
    def build_knowledgebase_faiss(self, text_chunks: list, **kwargs):
        self.logger.info(f"VectorStore Operation: build_knowledgebase_faiss")
        if kwargs.get('model') == 'huggingface':
            self.embedding_model = self.embeddings_huggingface
        else:
            self.embedding_model = self.embeddings_openai

        # Use embeddings for vectorization
        embeddings = self.embedding_model
        # Log the process of creating the vector database
        self.logger.info(f"Creating Vector Database using FAISS and {embeddings.__class__.__name__}")

        # Create vector database using FAISS with the embeddings
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        if kwargs.get("save") == True:
            self.logger.info(f"Saving Vector Database")
            vectorstore.save_local("faiss_index")
            self.logger.info(f"Loading Vector Database locally")
            vectorstore = FAISS.load_local("faiss_index", embeddings)
            return vectorstore
        return vectorstore

    def load_knowledgebase_from_disk_faiss(self, **kwargs):
        self.logger.info(f"VectorStore Operation: load_knowledgebase_from_disk_faiss")
        if kwargs.get('model') == 'huggingface':
            self.embedding_model = self.embeddings_huggingface
        else:
            self.embedding_model = self.embeddings_openai
        try:
            # Use embeddings for vectorization
            embeddings = self.embedding_model
            index_name = kwargs.get("index_name", "faiss_index")
            vectorstore = FAISS.load_local(index_name, embeddings)
            return vectorstore
        except Exception as e:
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception message: {e}")
            self.logger.error(f"Traceback: {traceback.print_exc()}")

    # %%
    def get_embeddings(self, text_chunks: list, **kwargs) -> list[list[float]]:
        """
                Generates embbedings using OpenAI embeddings default or HuggingFace models.

                :param metadata:
                :param text_chunks: List of text chunks to convert into embeddings.
                :param kwargs: Keyword arguments
                to pass to embedding model initialization :return: A tuple containing a list with vector store
                containing the vectorized representations of the input texts. and a list with text_chunks ids
        """
        self.logger.info(f"VectorStore Operation: get_embeddings")
        if kwargs.get('model') == 'huggingface':
            self.embedding_model = self.embeddings_huggingface
        else:
            self.embedding_model = self.embeddings_openai

        # Use embeddings for vectorization
        self.logger.info(f"Getting embeddings using {self.embedding_model.__class__.__name__}")
        embeddings = self.embedding_model.embed_documents(text_chunks)

        return embeddings

    # %%

    def build_knowledgebase_pinecone(self, documents: list, **kwargs) -> None:
        """
        Upsert Vectors into Pinecone Index: Now, you can upsert your vectors (embeddings) into the Pinecone index.

        :param documents: List of Document objects
        :param kwargs: Keyword arguments to pass to embedding model initialization
        :return: Return VectorStore initialized from documents and embeddings
        """
        self.logger.info(f"VectorStore Operation: build_knowledgebase_pinecone")

        if kwargs.get('model') == 'huggingface':
            self.embedding_model = self.embeddings_huggingface
        else:
            self.embedding_model = self.embeddings_openai

        #
        return None

    def get_vectorstore_pinecone(self, text: list[str], ids: list[str], **kwargs) -> Pinecone:

        self.logger.info(f"VectorStore Operation: get_vectorstore_pinecone")
        if kwargs.get('model') == 'huggingface':
            self.embedding_model = self.embeddings_huggingface
        else:
            self.embedding_model = self.embeddings_openai
        index_name = kwargs.get('pinecone_index_name', 'internal-knowledgebase')

        vectorstore = Pinecone.from_texts(
            ids=ids,
            batch_size=200,
            texts=text,
            index_name=index_name,
            embedding=self.embedding_model,
        )

        return vectorstore

    def similarity_search_pinecone(self, vectorstore: Pinecone, question: str):
        sim_search = vectorstore.similarity_search(
            query= question,# user question
            k=3 #return the most relevant docs
        )
        # Get the text from the results

        source_of_knowledgebase = "\n".join([x.page_content for x in sim_search])

        return source_of_knowledgebase

if __name__ == '__main__':
    txt = """FAISS FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for 
embeddings of multimedia documents that are similar to each other. It solves limitations of traditional query search 
engines that are optimized for hash-based searches, and provides more scalable similarity search functions."""
