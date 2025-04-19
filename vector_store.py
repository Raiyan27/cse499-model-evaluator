import os
import math
import time
import traceback
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document, BaseRetriever

import config
from pdf_processor import process_pdfs_in_directory

print(f"Vector Store: Using PERSIST_DIR={config.PERSIST_DIR}, COLLECTION_NAME={config.COLLECTION_NAME}")

def get_embedding_model(api_key: str) -> OpenAIEmbeddings | None:
    """Initializes the OpenAI Embeddings model."""
    try:
        print("Initializing Embeddings Model...")
        model = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            openai_api_key=api_key
        )
        print("Embeddings Model Initialized.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI Embeddings: {e}")
        traceback.print_exc()
        return None

def load_or_create_vector_db(embeddings_model: OpenAIEmbeddings) -> Chroma | None:
    """Loads an existing Chroma DB or creates a new one if necessary."""
    if not embeddings_model:
        print("ERROR: Embeddings model not available. Cannot load/create Vector DB.")
        return None

    db_exists = os.path.exists(config.PERSIST_DIR) and os.listdir(config.PERSIST_DIR)
    vector_db = None

    if db_exists:
        print(f"Attempting to load existing vector database from {config.PERSIST_DIR}...")
        try:
            start_time = time.time()
            vector_db = Chroma(
                collection_name=config.COLLECTION_NAME,
                persist_directory=config.PERSIST_DIR,
                embedding_function=embeddings_model
            )
            # Verify connection and count documents
            count = vector_db._collection.count()
            load_time = time.time() - start_time
            print(f"Vector database loaded successfully in {load_time:.2f}s with {count} documents.")
            return vector_db
        except Exception as e:
            print(f"ERROR: Error loading existing database: {e}. Will try to recreate.")
            print("WARNING: Manual deletion of the './chroma_db' directory might be required if recreation fails.")
            vector_db = None # Ensure it's None so we proceed to creation

    # If DB doesn't exist or loading failed, create it
    print(f"\nCreating new vector database in {config.PERSIST_DIR}...")
    if not os.path.exists(config.DATA_DIR):
         os.makedirs(config.DATA_DIR)
         print(f"WARNING: Created data directory: {config.DATA_DIR}. Please add PDF files there and rerun.")
         return None # Cannot create DB without data

    print("Processing PDF documents...")
    docs = process_pdfs_in_directory(config.DATA_DIR)
    if not docs:
        print("ERROR: No documents processed from PDF files. Cannot create vector database.")
        return None

    total_docs = len(docs)
    print(f"\nEmbedding {total_docs} document chunks (Batch size: {config.EMBEDDING_BATCH_SIZE})...")

    # Initialize Chroma collection first, without documents
    try:
        vector_db = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings_model,
            persist_directory=config.PERSIST_DIR
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize empty Chroma collection: {e}")
        traceback.print_exc()
        return None

    # Add documents in batches
    num_batches = math.ceil(total_docs / config.EMBEDDING_BATCH_SIZE)
    start_time_embed = time.time()

    try:
        for i in range(0, total_docs, config.EMBEDDING_BATCH_SIZE):
            batch_num = (i // config.EMBEDDING_BATCH_SIZE) + 1
            batch_docs = docs[i : i + config.EMBEDDING_BATCH_SIZE]

            if not batch_docs: continue

            batch_start_time = time.time()
            print(f"  Embedding batch {batch_num}/{num_batches} ({len(batch_docs)} docs)...", end="", flush=True)

            # Prepare IDs for the batch (optional but good practice)
            ids = [f"doc_{i+j}" for j in range(len(batch_docs))] # Simple unique IDs

            vector_db.add_documents(documents=batch_docs, ids=ids) # Provide IDs

            batch_end_time = time.time()
            print(f" done in {batch_end_time - batch_start_time:.2f}s.")

        total_embed_time = time.time() - start_time_embed
        print(f"\nEmbedding complete in {total_embed_time:.2f}s.")
        # Chroma with persist_directory usually handles persistence automatically.
        # Call persist() explicitly if experiencing issues, but often not needed.
        # print("Persisting database changes...")
        # vector_db.persist()
        # print("Persistence complete.")

        final_count = vector_db._collection.count()
        print(f"New vector database created and persisted with {final_count} documents.")
        return vector_db

    except Exception as e:
        print(f"\nERROR: Failed to create vector database during batch embedding: {e}")
        print(f"Error details: {traceback.format_exc()}")
        print(f"WARNING: Database creation failed. You might need to manually delete the '{config.PERSIST_DIR}' directory before retrying.")
        return None

def get_retriever(vector_db: Chroma) -> BaseRetriever | None:
    """Creates a retriever from the vector database."""
    if not vector_db:
        print("ERROR: Vector database is not available, cannot create retriever.")
        return None
    try:
        retriever = vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config.MAX_DOCS_PER_QUERY,
                "score_threshold": config.MINIMUN_RETRIVAL_SCORE
            }
        )
        print(f"Retriever created (k={config.MAX_DOCS_PER_QUERY}, threshold={config.MINIMUN_RETRIVAL_SCORE}).")
        return retriever
    except Exception as e:
        print(f"ERROR: Failed to create retriever: {e}")
        traceback.print_exc()
        return None

# Example usage (optional, for direct testing of this module)
if __name__ == "__main__":
    print("\nTesting Vector Store Module...")
    embeddings = get_embedding_model(config.OPENAI_API_KEY)
    if embeddings:
        db = load_or_create_vector_db(embeddings)
        if db:
            retriever = get_retriever(db)
            if retriever:
                print("\nTesting retrieval...")
                try:
                    # Perform a sample query
                    sample_query = "What are the requirements for company registration?"
                    results = retriever.invoke(sample_query)
                    print(f"Retrieved {len(results)} documents for query: '{sample_query}'")
                    # for doc in results:
                    #     print(f"  - Source: {doc.metadata.get('source', 'N/A')}, Score: {doc.metadata.get('_score', 'N/A')}") # Chroma might not add _score directly here
                except Exception as e:
                    print(f"ERROR during sample retrieval: {e}")
        else:
            print("Failed to load or create vector database.")
    else:
        print("Failed to initialize embedding model.")