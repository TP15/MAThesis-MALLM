import os
import json
import sys
import traceback
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Dict, List, Any, Optional

# --- Configuration ---

# Path to the single JSONL file containing combined/transformed data
INPUT_JSONL_FILE = "/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/final_RAG_MAL_CAPEC_DATA.jsonl" # ADJUST THIS PATH

# Directory where the FAISS index will be saved
FAISS_INDEX_PATH = "/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/capec_faiss_index" # Choose a new name for the combined index

# Name of the Sentence Transformer model to use for embeddings
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" # Or your preferred model

# --- End Configuration ---


# This function replaces the old load_and_prepare_docs
def load_docs_from_jsonl(jsonl_file_path: str) -> List[Document]:
    """
    Loads data from a JSONL file where each line has the pre-defined RAG structure,
    and creates LangChain Documents.

    Args:
        jsonl_file_path: Path to the input JSONL file.
                         Each line should be a JSON object like:
                         {
                           "embedding_input": "...",
                           "source_type": "MAL" | "CAPEC",
                           "metadata": { ... },
                           "raw": { ... } // raw is ignored here
                         }

    Returns:
        A list of LangChain Document objects.
    """
    all_docs: List[Document] = []
    print(f"Starting data loading from JSONL file: {jsonl_file_path}")

    if not os.path.exists(jsonl_file_path):
        print(f"Error: Input file not found: {jsonl_file_path}")
        return [] # Return empty list if file doesn't exist

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                try:
                    entry = json.loads(line)

                    # Extract data based on the expected structure
                    page_content = entry.get("embedding_input")
                    metadata = entry.get("metadata", {}) # Get metadata dict, default to empty
                    source_type = entry.get("source_type") # Get source type

                    # Ensure essential data is present
                    if not page_content:
                        print(f"Warning: Skipping line {line_num} due to missing 'embedding_input'.")
                        continue
                    if not metadata:
                         print(f"Warning: Line {line_num} has missing 'metadata'. Using empty metadata.")
                    if source_type:
                        # Add source_type to the metadata dict for potential filtering later
                        metadata['source_type'] = source_type
                    else:
                        print(f"Warning: Line {line_num} has missing 'source_type'. It won't be added to metadata.")


                    # --- Create LangChain Document ---
                    # Use 'embedding_input' directly as the content to be embedded.
                    # Use the 'metadata' dictionary directly from the JSONL entry.
                    doc = Document(page_content=page_content, metadata=metadata)
                    all_docs.append(doc)

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {line[:100]}...")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}. Data: {line[:100]}...")

    except IOError as e:
        print(f"Error reading file {jsonl_file_path}: {e}")
        return [] # Return empty list on file read error
    except Exception as e:
         print(f"An unexpected error occurred during file processing: {e}")
         return []


    print(f"Finished loading. Total documents prepared: {len(all_docs)}")
    return all_docs


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting RAG Phase 1: Indexing Combined MAL/CAPEC Data ---")
    print(f"Input JSONL file: {INPUT_JSONL_FILE}")
    print(f"Vector store persistence directory: {FAISS_INDEX_PATH}")
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")

    # 1. Load documents from the single JSONL file
    documents = load_docs_from_jsonl(INPUT_JSONL_FILE)

    if not documents:
        print("\nNo documents were loaded. Please check the input JSONL file exists and contains valid data. Exiting.")
        sys.exit(1) # Use sys.exit for clearer exit status

    # 2. Initialize embedding model
    print(f"\nInitializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    # model_kwargs = {'device': 'cpu'} # Uncomment to force CPU if needed
    encode_kwargs = {'normalize_embeddings': False}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            # model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print("Make sure 'sentence-transformers' and potentially 'torch' are installed correctly.")
        sys.exit(1)

    # 3. Create and persist the FAISS vector store
    print(f"\nCreating FAISS index and saving to: {FAISS_INDEX_PATH}")
    try:
        # Calculate embeddings and create FAISS index
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )

        # Save the index and document store locally
        vectorstore.save_local(folder_path=FAISS_INDEX_PATH)

        print(f"\n--- Success! ---")
        print(f"FAISS index created and saved successfully in '{FAISS_INDEX_PATH}'.")
        print(f"Total documents indexed: {len(documents)}")

        # Optional: Simple test query requires loading the index first
        print("\nPerforming a quick test query (loading from disk)...")
        if not os.path.exists(FAISS_INDEX_PATH):
             print(f"  Error: Saved index path '{FAISS_INDEX_PATH}' not found for testing.")
        else:
            try:
                # Load the persisted index for testing
                loaded_vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True # Required by recent LangChain versions
                )
                # Example query - adjust based on your data (MAL or CAPEC)
                test_query = "Hardware supply chain attack description"
                results = loaded_vectorstore.similarity_search(test_query, k=1)

                if results:
                    print(f"  Test query '{test_query}' found result:")
                    # Access metadata directly from the loaded document's metadata dict
                    # Check the keys that exist in your combined_rag_data.jsonl's metadata
                    meta = results[0].metadata
                    source_type = meta.get('source_type', 'N/A')
                    doc_name = meta.get('name', meta.get('mal_type', 'N/A')) # Try 'name' (CAPEC) or 'mal_type' (MAL)
                    capec_id = meta.get('capec_id', 'N/A') # Check if CAPEC ID exists
                    print(f"    Source Type: {source_type}")
                    print(f"    Name/Type: {doc_name}")
                    if capec_id != 'N/A':
                         print(f"    CAPEC ID: {capec_id}")
                    # print(f"    Content Snippet: {results[0].page_content[:150]}...") # Uncomment to see content
                else:
                    print(f"  Test query '{test_query}' returned no results.")
            except Exception as e:
                print(f"  Error during test query loading/execution: {e}")
                traceback.print_exc() # Print traceback for loading errors


    except Exception as e:
        print(f"\n--- Error ---")
        print(f"An error occurred during FAISS index creation/saving: {e}")
        traceback.print_exc() # Print full traceback for FAISS errors

    print("\n--- RAG Phase 1 Finished (Using FAISS) ---")