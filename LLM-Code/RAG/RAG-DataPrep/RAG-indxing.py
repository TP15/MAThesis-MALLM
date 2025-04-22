import os
import json
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Dict, List, Any, Optional

# --- Configuration ---

# Directory where the JSON files from the previous step are located
INPUT_DATA_DIR = "capec_rag_input_data"
# Directory where the FAISS index will be saved
FAISS_INDEX_PATH = "capec_faiss_index" # Name des Ordners für den FAISS Index

# Name of the Sentence Transformer model to use for embeddings
# 'all-MiniLM-L6-v2' is a good starting point: fast and decent quality.
# Other options: 'all-mpnet-base-v2' (better quality, slower), etc.
# Make sure you have enough RAM/disk space for the chosen model.
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Object types to process from the input directory
OBJECT_TYPES_TO_PROCESS = ["attack-pattern", "course-of-action"]

# --- End Configuration ---

def get_capec_id(capec_obj_dict: Dict[str, Any]) -> Optional[str]:
    """
    Extracts the CAPEC ID from a dictionary representing a STIX object.

    Args:
        capec_obj_dict: The dictionary loaded from the JSON file.

    Returns:
        The CAPEC ID (e.g., "CAPEC-66") or None if not found.
    """
    ext_refs = capec_obj_dict.get("external_references", [])
    for ref in ext_refs:
        if ref.get('source_name') == 'capec' and ref.get('external_id'):
            ext_id = ref['external_id']
            # Handle potential variations in ID format within the JSON
            if isinstance(ext_id, int):
                 return f"CAPEC-{ext_id}"
            elif isinstance(ext_id, str):
                 # Ensure it has the prefix, handles cases like "66" or "CAPEC-66"
                 return ext_id if ext_id.startswith("CAPEC-") else f"CAPEC-{ext_id}"
    return None

def load_and_prepare_docs(input_dir: str, object_types: List[str]) -> List[Document]:
    """
    Loads data from JSON files, extracts text and metadata, and creates LangChain Documents.

    Args:
        input_dir: Directory containing the '*_rag_data.json' files.
        object_types: List of STIX object types (filenames prefixes) to load.

    Returns:
        A list of LangChain Document objects.
    """
    all_docs: List[Document] = []
    print(f"Starting data loading and preparation from: {input_dir}")

    for obj_type in object_types:
        file_path = os.path.join(input_dir, f"{obj_type}_rag_data.json")
        print(f"  Processing file: {file_path}")

        if not os.path.exists(file_path):
            print(f"    Warning: File not found, skipping.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"    Loaded {len(data)} objects.")

            for capec_obj_dict in data:
                # --- Extract and Combine Text Fields ---
                # Select the most relevant fields for semantic understanding.
                # Handle potentially missing fields gracefully with .get(key, default).
                name = capec_obj_dict.get('name', '')
                desc = capec_obj_dict.get('description', '')
                # Examples can be a list or string, handle list case
                examples_list = capec_obj_dict.get('x_capec_example_instances', [])
                examples_str = "\n".join(examples_list) if isinstance(examples_list, list) else str(examples_list)
                # Execution flow might be HTML or complex string, keep as is for now
                exec_flow = capec_obj_dict.get('x_capec_execution_flows', '')
                prereqs_list = capec_obj_dict.get('x_capec_prerequisites', [])
                prereqs_str = "\n".join(prereqs_list) if isinstance(prereqs_list, list) else str(prereqs_list)

                # Combine into a single text block for the document content
                page_content = (
                    f"CAPEC Name: {name}\n\n"
                    f"Type: {capec_obj_dict.get('type', '')}\n\n"
                    f"Description: {desc}\n\n"
                    f"Example Instances:\n{examples_str}\n\n"
                    f"Execution Flow:\n{exec_flow}\n\n"
                    f"Prerequisites:\n{prereqs_str}"
                    # Add other relevant fields like 'x_capec_typical_severity', 'x_capec_skills_required' if needed
                ).strip() # Remove leading/trailing whitespace

                # --- Extract Metadata ---
                capec_id = get_capec_id(capec_obj_dict)
                metadata = {
                    "stix_id": capec_obj_dict.get('id', 'N/A'),
                    "capec_id": capec_id if capec_id else 'N/A',
                    "object_type": capec_obj_dict.get('type', 'N/A'),
                    "name": name,
                    # Add source file info for traceability
                    "source_file": os.path.basename(file_path)
                }

                # Create LangChain Document
                doc = Document(page_content=page_content, metadata=metadata)
                all_docs.append(doc)

        except json.JSONDecodeError as e:
            print(f"    Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            print(f"    An unexpected error occurred while processing {file_path}: {e}")

    print(f"Finished loading. Total documents prepared: {len(all_docs)}")
    return all_docs


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting RAG Phase 1: Indexing CAPEC Data ---")
    print(f"Input data directory: {INPUT_DATA_DIR}")
    print(f"Vector store persistence directory: {FAISS_INDEX_PATH}")
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")

    # 1. Load and prepare documents
    documents = load_and_prepare_docs(INPUT_DATA_DIR, OBJECT_TYPES_TO_PROCESS)

    if not documents:
        print("\nNo documents were loaded. Please check the input directory and file contents. Exiting.")
        exit(1)

    # 2. Initialize embedding model
    print(f"\nInitializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    # You can specify device='cuda', device='mps' etc. if needed and available
    # model_kwargs = {'device': 'cpu'} # Uncomment to force CPU
    encode_kwargs = {'normalize_embeddings': False} # Usually good practice for similarity search
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            # model_kwargs=model_kwargs, # Specify device if needed
            encode_kwargs=encode_kwargs
        )
        print("Embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print("Make sure 'sentence-transformers' and potentially 'torch' are installed correctly.")
        exit(1)

# --- MODIFIED BLOCK START ---
# 3. Create and persist the FAISS vector store
print(f"\nCreating FAISS index and saving to: {FAISS_INDEX_PATH}")
try:
    # This single command does the following:
    # - Calculates embeddings for all documents (can take time!)
    # - Creates a FAISS index in memory
    # - Adds the documents and embeddings to the index
    # NOTE: Ensure 'documents' contains your list of LangChain Document objects
    # NOTE: Ensure 'embeddings' is your initialized HuggingFaceEmbeddings object
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Save the index and document store locally
    vectorstore.save_local(folder_path=FAISS_INDEX_PATH)

    print(f"\n--- Success! ---")
    print(f"FAISS index created and saved successfully in '{FAISS_INDEX_PATH}'.")
    print(f"Total documents indexed: {len(documents)}") # Use count from input docs

    # Optional: Simple test query requires loading the index first
    print("\nPerforming a quick test query (loading from disk)...")
    try:
        # Load the persisted index for testing
        # IMPORTANT: allow_dangerous_deserialization=True is often needed for FAISS loading
        # due to internal use of pickle. Only load indexes you trust.
        loaded_vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True # Required by recent LangChain versions
        )
        test_query = "SQL Injection techniques"
        results = loaded_vectorstore.similarity_search(test_query, k=1)
        if results:
            print(f"  Test query '{test_query}' found result with CAPEC ID: {results[0].metadata.get('capec_id', 'N/A')}")
        else:
            print(f"  Test query '{test_query}' returned no results.")
    except Exception as e:
        print(f"  Error during test query loading/execution: {e}")
        import traceback
        traceback.print_exc() # Print traceback for loading errors


except Exception as e:
    print(f"\n--- Error ---")
    print(f"An error occurred during FAISS index creation/saving: {e}")
    import traceback
    traceback.print_exc() # Print full traceback for FAISS errors

# --- MODIFIED BLOCK END ---

print("\n--- RAG Phase 1 Finished (Using FAISS) ---")