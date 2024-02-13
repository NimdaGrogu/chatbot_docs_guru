import time
from pinecone import Index, Pinecone
from langchain_pinecone import Pinecone
from langchain_community.vectorstores import Pinecone

# from dotenv import load_dotenv, find_doten
import pinecone
import os
import json
import logging


class PineconeOperations(Pinecone):

    def __init__(self):
        # Create a logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Picone Operation: Initiation {self.__class__.__name__}")
        self.index = None
        #  Instance of the Parent class and get ready methods and attributes
        #super().__init__()

    def create_index(self, **kwargs) -> None:
        index_name = kwargs.get('pinecone_index_name', 'internal-knowledgebase')
        # Pinecone Dimension OpenAIEmbeddigs default dimension other dimension # dimension = 768
        pinecone_dimension = kwargs.get('pinecone_dimension', 1536)  # 1536 dim of text-embedding-ada-002
        # Pinecone metric
        pinecone_metric = kwargs.get("pinecone_metric", "cosine")
        self.logger.info(f"Picone Operation: Creating index if it doesn't exist")
        # create index if there are no indexes found
        # Create a pinecone index if not exist
        #  Ensure  the correct EMBEDDING_DIMENSION that
        #  matches the output dimension of the embedding model.

        if index_name not in pinecone.list_indexes().names():
            pinecone.create_index(name=index_name, dimension=pinecone_dimension, metric=pinecone_metric)
            self.logger.info("Wait for index to finish initialization")
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(2)
            self.logger.info(f"{pinecone.describe_index(index_name).status['ready']}")
        else:
            self.logger.info(f"Pinecone Operation:  Index {index_name} found")

        return None

    def connect_index(self, **kwargs) -> Index:
        index_name = kwargs.get('pinecone_index_name', 'internal-knowledgebase')
        # connect to a specific index
        # client for interacting with a Pinecone index via Index.
        self.index = self.Index(name=index_name)
        self.logger.info(f"Picone Operation: Connection to index \n{self.fetch_stats()} ")
        return self.index  # A client for interacting with a Pinecone index via API.

    def upsert(self, data):
        # sample data of the format
        # [
        #     ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        #     ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        #     ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
        #     ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
        #     ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # ]
        # Upsert sample data (5 8-dimensional vectors)
        return json.loads(str(self.index.upsert(vectors=data, namespace="quickstart")).replace("'", '"'))

    def fetch_stats(self):
        # fetches stats about the index
        self.logger.info(f"Picone Operation: Fetching stats")

        stats = self.index.describe_index_stats()
        return str(stats)

    def query(self, query_vector):
        # query from the index, eg: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        response = self.index.query(
            vector=query_vector,
            top_k=2,
            include_values=True,
            namespace="quickstart"
        )
        return json.loads(str(response).replace("'", '"'))

    def embed_from_docs(self, documents: list, embedding_model) -> None:
        # constructs a Pinecone wrapper from raw documents
        # This method is a user-friendly interface that embeds documents and
        # adds them to a provided Pinecone index.
        # The method returns an instance of the Pinecone class.
        self.logger.info(f"Picone Operation: Embedding from Docs \n{len(documents)}")

        Pinecone.from_documents(documents=documents,
                                embeddings=embedding_model,
                                index_name = self.index.name)



        return None

    def get_vectorstore(self, embedding_model, metadata_text_field: str) -> Pinecone:
        ## Connect to the index
        pinecone_client = self.connect_index()
        self.logger.info(f"Picone Operation: get_vectorstore ")
        text_field = metadata_text_field  # the metadata field that contains our text
        vectorstore = Pinecone(
            pinecone_client,
            embedding_model,
            text_key=text_field
        )
        return vectorstore
