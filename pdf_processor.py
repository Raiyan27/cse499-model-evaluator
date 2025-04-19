import os
import fitz  # PyMuPDF
import traceback
import sys
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import config # Use constants from config.py

print(f"PDF Processing: Using DATA_DIR={config.DATA_DIR}")

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extracts text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"ERROR: Error processing {pdf_path}: {e}")
        traceback.print_exc()
        return None

def process_pdfs_in_directory(directory_path: str) -> List[Document]:
    """Processes all PDFs in a directory, extracts text, and splits into documents."""
    all_docs: List[Document] = []
    if not os.path.exists(directory_path):
        print(f"WARNING: Directory not found: {directory_path}")
        return []

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"WARNING: No PDF files found in {directory_path}")
        return []

    total_files = len(pdf_files)
    print(f"Processing {total_files} PDF files from {directory_path}...")

    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(directory_path, filename)
        print(f"Processing: {filename} ({i+1}/{total_files})...", end="", flush=True)
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            # Consider adjusting chunk size/overlap based on testing needs
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            # Create documents with source metadata
            docs = text_splitter.create_documents([pdf_text], metadatas=[{"source": filename}])
            all_docs.extend(docs)
            print(f" extracted {len(docs)} chunks.")
        else:
            print(" failed to extract text.")

    if not all_docs:
        print("WARNING: No text could be extracted from any PDF files.")
    else:
        print(f"Finished processing {total_files} PDF files, extracted {len(all_docs)} document chunks.")

    return all_docs

# Example usage (optional, for direct testing of this module)
if __name__ == "__main__":
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        print(f"Created data directory: {config.DATA_DIR}. Please add PDF files there.")
    else:
        processed_docs = process_pdfs_in_directory(config.DATA_DIR)
        print(f"\nTotal documents processed: {len(processed_docs)}")
        # if processed_docs:
        #     print("\nSample Document:")
        #     print(processed_docs[0])