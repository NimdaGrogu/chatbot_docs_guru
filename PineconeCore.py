import time
from pinecone import Index, Pinecone, PodSpec
from tqdm.auto import tqdm
# from langchain_community.vectorstores import Pinecone as VectorStorePinecone
# from dotenv import load_dotenv, find_dotenv
import os
import json
import logging


class PineconeOperations(Pinecone):

    def __init__(self):
        # Create a logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Picone Operation: Initiation {self.__class__.__name__}")

        #  Instance of the Parent class and get ready methods and attributes
        super().__init__()

    def create_pinecone_index(self, **kwargs) -> None:
        index_name = kwargs.get('pinecone_index_name', 'internal-knowledgebase')
        # Pinecone Dimension OpenAIEmbeddigs default dimension other dimension # dimension = 768
        pinecone_dimension = kwargs.get('pinecone_dimensioddn', 1536)  # 1536 dim of text-embedding-ada-002
        # Pinecone metric
        pinecone_metric = kwargs.get("pinecone_metric", "cosine")

        self.logger.info(f"Picone Operation: Creating index if it doesn't exist")
        # Delete the Index if already exist
        if index_name in [index.name for index in self.list_indexes()]:
            self.delete_index(index_name)

        # create index if there are no indexes found
        # Create a pinecone index if not ex11ist
        #  Ensure  the correct EMBEDDING_DIMENSION that
        #  matches the output dimension of the embedding model.
        if index_name not in self.list_indexes().names():
            self.logger.info(f"Index {index_name} not found in indexes list")
            self.logger.info(f"Creating Index {index_name}, Wait for index to finish initialization")
            environment = os.environ.get('PINECONE_ENVIRONMENT')
            self.create_index(name=index_name,
                              dimension=pinecone_dimension,
                              metric=pinecone_metric,
                              spec=PodSpec(environment=environment)
                              )
            while not self.describe_index(index_name).status['ready']:
                time.sleep(2)
        else:
            self.logger.info(f"Pinecone Operation:  Index {index_name} found")

        return None

    def connect_index(self, **kwargs) -> Index:
        index_name = kwargs.get('pinecone_index_name', 'internal-knowledgebase')
        # connect to a specific index
        # client for interacting with a Pinecone index via Index.
        self.index = self.Index(name=index_name)
        self.logger.info(f"Picone Operation: Connection to index ")
        return self.index  # A client for interacting with a Pinecone index via API.

    def fetch_stats(self) -> None:
        # fetches stats about the index
        self.logger.info(f"Picone Operation: Fetching stats")
        self.logger.info(f"{self.index.describe_index_stats()}")

    def pinecone_upsert(self, vectors: list, ids: list, pdf_metadata: str, batch_size=200):
        prepped = []
        index = self.connect_index()
        self.logger.info(f"Picone Operation: pinecone_upsert ")
        vect_ids = zip(vectors, ids)
        for vector, id in tqdm(vect_ids, total=len(vectors), desc="Upserting data"):
            prepped.append({'id': id,
                            'values': vector,
                            'metadata': pdf_metadata}
                           )
            if len(prepped) >= batch_size:
                index.upsert(prepped)
                prepped = []

        time.sleep(2)
        self.fetch_stats()
