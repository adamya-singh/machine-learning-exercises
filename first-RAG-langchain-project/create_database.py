from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_pinecone import Pinecone as PineconeVectorStore
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone as PineconeClent, ServerlessSpec
from dotenv import load_dotenv
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/current_data"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def main():
    #only runs if the file is run directly, not if called as a module
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    generate_data_store(openai_api_key, pinecone_api_key)

def generate_data_store(openai_api_key, pinecone_api_key):
    documents = load_documents()
    chunks = split_text(documents)
    #save_to_chroma(chunks, openai_api_key)
    save_to_pinecone(chunks, openai_api_key, pinecone_api_key)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

load_documents()

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 50,
        length_function = len,
        add_start_index = True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    #document = chunks[0]
    #print(document.page_content)
    #print(document.metadata)

    return chunks
"""
def save_to_chroma(chunks: list[Document], openai_api_key):
    #remove existing database if present
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    #create a new database from the documents    
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(openai_api_key=openai_api_key),
        persist_directory = CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
"""

def save_to_pinecone(chunks: list[Document], openai_api_key, pinecone_api_key):
    #create a pinecone client instance
    pc = PineconeClent(pinecone_api_key)
    
    #name of the vector database on pinecone website
    index_name = "rutgers-menu-assistant"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    index = pc.Index(index_name)

    #embed the documents and save to pinecone
    #embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")

    db = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name,
    )

    print(f"Saved {len(chunks)} chunks to Pinecone index {index_name}.")

if __name__ == "__main__":
    #makes sure main() only runs when this script is called directly
    #not when called as a module (when imported into another script)
    main()