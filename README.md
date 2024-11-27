# rag-pilot
A starter project for building end-to-end RAG pipeline working on domain-specific applications

## Setup
Ensure all dependencies are installed:
-> pip install langchain chromadb langchain-community

## Prepare the vector database:

1. Populate the Chroma database (CHROMA_PATH) with documents and their embeddings using the OllamaEmbeddings.
2. Run the Application. Execute the script by passing a query as an argument:
-> python script.py "Your query text here"
