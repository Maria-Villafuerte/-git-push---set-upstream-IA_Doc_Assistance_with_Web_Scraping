from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from consts import INDEX_NAME
import os
import sys
import io
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    try:
        # Pinecone initialization checks (omitted for brevity)

        loader = DirectoryLoader(
            "../android_docs",
            glob="**/*.html",
            show_progress=True,
            loader_cls=UnstructuredHTMLLoader
        )
        logging.info("Starting to load documents...")
        raw_documents = loader.load()
        logging.info(f"Loaded {len(raw_documents)} raw documents")

        if not raw_documents:
            logging.warning("No documents were loaded. Check if the directory contains HTML files.")
            return

        # Log the content of the first document for debugging
        if raw_documents:
            logging.info(f"First document content (truncated): {raw_documents[0].page_content[:500]}...")

        # Adjust text splitter parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logging.info("Starting to split documents...")
        documents = text_splitter.split_documents(raw_documents)
        logging.info(f"Split into {len(documents)} chunks")

        if not documents:
            logging.warning("No document chunks were created after splitting.")
            # Log more details about the raw documents
            for i, doc in enumerate(raw_documents):
                logging.info(f"Raw document {i} length: {len(doc.page_content)}")
                logging.info(f"Raw document {i} content preview: {doc.page_content[:100]}...")
            return

        # Log the content of the first split chunk for debugging
        if documents:
            logging.info(f"First chunk content (truncated): {documents[0].page_content[:500]}...")

        if not raw_documents:
            logging.warning("No documents were loaded. Check if the directory contains HTML files.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_documents)
        logging.info(f"Split into {len(documents)} chunks")

        if not documents:
            logging.warning("No document chunks were created after splitting.")
            return

        for doc in documents:
            relative_path = os.path.relpath(doc.metadata["source"], "../android_docs")
            doc.metadata["source"] = f"https://developer.android.com/{relative_path}"

        logging.info(f"Going to add {len(documents)} to Pinecone")

        try:
            pinecone_index = PineconeVectorStore.from_documents(
                documents, embedding=embeddings, index_name=INDEX_NAME
            )
            logging.info("Successfully created PineconeVectorStore instance")
        except Exception as e:
            logging.error(f"Error creating PineconeVectorStore: {str(e)}")
            return

        for i, doc in enumerate(documents):
            try:
                pinecone_index.add_documents([doc])
                logging.info(f"File uploaded: {doc.metadata['source']} ({i + 1}/{len(documents)})")
            except Exception as e:
                logging.error(f"Error uploading file {doc.metadata['source']}: {str(e)}")

        logging.info("Document ingestion process completed.")

    except ImportError as ie:
        logging.error(f"Import error: {str(ie)}")
        logging.info("Please ensure all required packages are installed.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    ingest_docs()