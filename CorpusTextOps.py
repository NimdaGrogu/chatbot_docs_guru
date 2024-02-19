# Import necessary libraries
import os
import subprocess
import logging
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Basic configuration of logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_pdf_text_metadata(pdf_docs) -> tuple:
    """
    Extracts text from a list of PDF file Obj in binary mode.

    :param pdf_docs: A list of PDF document.
    :return: A string containing the concatenated text of all the PDF documents.
    """
    text = ""
    metadata = {}

    for pdf in pdf_docs:
        # Initialize a PDF reader for each document
        pdf_reader = PdfReader(pdf)

        # Extract and clean Metadata from the PDF
        file_metadata = dict(pdf_reader.metadata)
        for k, v in file_metadata.items():
            k = k.replace('/', '')
            metadata.update({f"{k}": f"{v}"})

        # Iterate through each page in the PDF
        for page in pdf_reader.pages:
            # Extract text from the page and append it to the text variable
            text += page.extract_text()

    metadata = [metadata]
    return text, metadata


def get_word_text():
    """
    Placeholder function for future implementation.
    Intended to extract text from Word documents.
    """
    pass


def get_text_chunks_ids(text):
    """
    Splits a given text into smaller chunks and creates LangChain documents.

    :param text: The text to be split.
    :return: A tuple containing two elements;
             1. List of text chunks
             2. List of LangChain 'Documents' created from the text chunks
    """
    logging.info("Initialize the text splitter with specific parameters")
    # Initialize the text splitter with specific parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # The maximum size of each chunk
        chunk_overlap=0,  # Overlap between chunks to retain context
        length_function=len  # Function to measure the length of text
    )
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    logging.info(f"Chunks: {len(chunks)}")
    # These IDs are necessary for indexing and querying the vectors.
    # Create Vector IDs: Create unique IDs for each of your text chunks.

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    return chunks, ids


def read_pdf_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    pdf_docs = file_loader.load()
    return pdf_docs


def identify_file_type(file_path):
    # Create a logger for this method
    logger = logging.getLogger("identify_file_type")
    logger.info(f"identify_file_type")
    allowed_files = ['pdf', 'PDF', 'docx', 'text']
    # Use the 'file' command to identify the file type
    result = subprocess.run(['file', '--mime-type', '--brief', file_path], capture_output=True, text=True, check=True)
    if result.returncode == 0:
        file_type = result.stdout.strip()
        for file in allowed_files:
            if file in file_type:
                logger.info(f"File type is {file_type}")
                return file_type
            logger.error(f"File type is not allowed {file_path}")
            return None
