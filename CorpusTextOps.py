# Import necessary libraries
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF file Obj in binary mode.

    :param pdf_docs: A list of PDF document paths.
    :return: A string containing the concatenated text of all the PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        # Initialize a PDF reader for each document
        pdf_reader = PdfReader(pdf)
        # Iterate through each page in the PDF
        for page in pdf_reader.pages:
            # Extract text from the page and append it to the text variable
            text += page.extract_text()
    return text


def get_word_text():
    """
    Placeholder function for future implementation.
    Intended to extract text from Word documents.
    """
    pass


def get_text_chunks(text):
    """
    Splits a given text into smaller chunks and creates LangChain documents.

    :param text: The text to be split.
    :return: A tuple containing two elements;
             1. List of text chunks
             2. List of LangChain 'Documents' created from the text chunks
    """
    # Initialize the text splitter with specific parameters
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Define the separator for splitting
        chunk_size=800,  # The maximum size of each chunk
        chunk_overlap=50,  # Overlap between chunks to retain context
        length_function=len  # Function to measure the length of text
    )
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    # Create LangChain documents from the text chunks
    langchain_docs = text_splitter.create_documents(text)
    return chunks, langchain_docs


def read_pdf_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    pdf_docs = file_loader.load()
    return pdf_docs


def chunk_data(docs, chunk_size=800, chunk_overlap=50) -> list:

    # Initialize the text splitter with specific parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # The maximum size of each chunk
        chunk_overlap=chunk_overlap,  # Overlap between chunks to retain context
    )
    docs = text_splitter.split_documents(documents=docs)
    return docs
