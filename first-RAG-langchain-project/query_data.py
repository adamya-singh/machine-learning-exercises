# query.py
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_pinecone import PineconeVectorStore
#from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient

# If you want to reuse code from create_database.py, you might do something like:
# from create_database import load_documents, split_text, save_to_pinecone
# But note you'd need to refactor create_database.py to ensure imports and references work cleanly.
# For now, we'll just assume the index already exists.

def main():
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    index_name = "rutgers-menu-assistant"

    # Initialize Pinecone client
    pc = PineconeClient(pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"Index '{index_name}' does not exist. Run create_database.py first.")

    index = pc.Index(index_name)
    print("Total vectors in the index:", index.describe_index_stats())

    # Use the same embeddings as during indexing for consistency
    embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")
    
    # Create a vector store from the existing Pinecone index
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key='text'
    )

    # Create the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Use a locally running model via Ollama
    # Make sure you have ollama installed and a model pulled, e.g. `ollama pull llama2`
    llm = OllamaLLM(
        model="llama3.2:3b",  # Replace with the name of your local model
        base_url="http://127.0.0.1:11434"  # default ollama server endpoint
    )

    # Create a RetrievalQA chain (simplest approach)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Query your vector store using local Ollama LLM
    user_query = "What breakfast items are listed?"
    answer = qa_chain.invoke({"query": user_query})
    print("Answer:", answer)

if __name__ == "__main__":
    main()
