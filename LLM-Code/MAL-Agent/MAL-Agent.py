import os
import subprocess
import tempfile
import json
import torch
import logging
import re
import numpy as np
import pickle
import faiss
import shlex
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "./path/to/your/qlora_adapter" # TODO: Update this path

# --- IMPORTANT: MAL Compiler Path ---
# Update this path to point to the ACTUAL location of the 'malc' executable
# that you extracted from the .tar.gz file (e.g., './malc_extracted/malc')
# or the location where it was installed by the .deb package (e.g., '/usr/local/bin/malc' or just 'malc' if in PATH)
MALC_EXECUTABLE_PATH = './malc_extracted/malc' # TODO: UPDATE THIS PATH!

# --- RAG Configuration ---
EMBEDDING_MODEL_ID = 'all-MiniLM-L6-v2'
VECTOR_STORE_PATH = "capec_faiss_index" # TODO: Update or ensure this exists
USE_FAISS = True # Assumes FAISS index exists at VECTOR_STORE_PATH
TOP_K_RAG = 4

# --- LLM Configuration ---
MAX_ATTEMPTS = 5
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_P = 0.9
DO_SAMPLE = True

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Prompt Templates ---
INITIAL_PROMPT_TEMPLATE = """
Hier ist relevanter Kontext aus der CAPEC-Datenbank über Angriffsmuster:
---
{context}
---
Basierend auf diesem Kontext und der folgenden Beschreibung, generiere bitte validen MAL-Code.
Beschreibung: {input_text}

MAL Code:
"""

REFINEMENT_PROMPT_TEMPLATE = """
Der vorherige MAL-Code war nicht korrekt. Compiler-Fehler:
{compiler_error}

Hier ist der ursprüngliche Kontext aus der CAPEC-Datenbank:
---
{context}
---
Hier ist die ursprüngliche Beschreibung: {input_text}
Hier ist der fehlerhafte MAL-Code:
{previous_code}

Bitte korrigiere den MAL-Code basierend auf dem Compiler-Feedback, dem Kontext und der Beschreibung.

Korrigierter MAL Code:
"""

def setup_rag(embedding_model_id, index_folder_path):
    """Sets up RAG components by loading FAISS index and docstore."""
    logging.info(f"Setting up RAG by loading index from: {index_folder_path}")
    faiss_index_file = os.path.join(index_folder_path, "index.faiss")
    docstore_file = os.path.join(index_folder_path, "index.pkl")

    if not (USE_FAISS and os.path.exists(faiss_index_file) and os.path.exists(docstore_file)):
        logging.warning(f"FAISS index ('index.faiss') or docstore ('index.pkl') not found in {index_folder_path} or USE_FAISS is False. RAG will be disabled.")
        return None, None, None

    try:
        faiss_index = faiss.read_index(faiss_index_file)
        logging.info(f"FAISS index loaded with {faiss_index.ntotal} vectors.")

        with open(docstore_file, "rb") as f:
            docstore_data = pickle.load(f)
            logging.info("Docstore data loaded.")

        # Handle potential FAISS index/docstore format variations (common issue)
        if isinstance(docstore_data, dict): # Simple dict mapping index to doc
             docstore_map = docstore_data
        elif isinstance(docstore_data, tuple) and len(docstore_data) == 2 and isinstance(docstore_data[0], dict) and isinstance(docstore_data[1], dict):
             # Format from langchain FAISS save_local: (index_to_docstore_id, docstore)
             index_to_docstore_id = docstore_data[0]
             actual_docstore = docstore_data[1]
             # Reconstruct a simple integer index -> document map
             docstore_map = {i: actual_docstore[index_to_docstore_id[str(i)]] for i in range(faiss_index.ntotal) if str(i) in index_to_docstore_id}
             logging.info(f"Reconstructed docstore map from Langchain FAISS format. Found {len(docstore_map)} mappings.")
        else:
             logging.error(f"Unexpected format ({type(docstore_data)}) loaded from docstore file '{docstore_file}'. Cannot build docstore map.")
             return None, None, None

        if not docstore_map:
             logging.error("Docstore map is empty after loading. Check index/docstore integrity.")
             return None, None, None

        embedding_model = SentenceTransformer(embedding_model_id)
        logging.info("RAG embedding model loaded.")
        return faiss_index, docstore_map, embedding_model

    except Exception as e:
        logging.error(f"RAG setup error: {e}", exc_info=True)
        return None, None, None

def get_rag_examples(query_text, embedding_model, faiss_index, docstore_map, top_k):
    """Retrieves relevant documents from FAISS index for a given query."""
    if not (USE_FAISS and faiss_index and docstore_map and embedding_model):
        # logging.debug("RAG components not available or disabled. Skipping retrieval.")
        return []

    try:
        logging.info(f"Encoding query for RAG: '{query_text[:100]}...'")
        query_embedding = embedding_model.encode([query_text], normalize_embeddings=True)
        logging.info(f"Searching FAISS index with top_k={top_k}")
        distances, indices = faiss_index.search(query_embedding, top_k)

        retrieved_contents = []
        logging.info(f"Retrieved indices: {indices[0]}")
        for i in indices[0]:
            if i != -1 and i in docstore_map:
                document = docstore_map[i]
                # Check if document has 'page_content' attribute (like Langchain docs)
                if hasattr(document, 'page_content'):
                    retrieved_contents.append(document.page_content)
                elif isinstance(document, str): # Handle simple string docs
                     retrieved_contents.append(document)
                else:
                     logging.warning(f"Document at index {i} has unexpected type {type(document)} and no 'page_content'. Skipping.")
            elif i == -1:
                 logging.debug(f"FAISS search returned -1 index.")
            else:
                 logging.warning(f"Index {i} from FAISS search not found in the loaded docstore map.")

        logging.info(f"Retrieved {len(retrieved_contents)} documents for RAG context.")
        return retrieved_contents
    except Exception as e:
        logging.error(f"RAG retrieval error: {e}", exc_info=True)
        return []

def run_llm_inference(prompt, model, tokenizer):
    """Runs inference using the loaded LLM and tokenizer."""
    logging.info(f"Running LLM inference. Prompt length: {len(prompt)}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id # Important for open-ended generation
        )
    # Decode only the newly generated tokens, excluding the prompt
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    logging.info(f"LLM Inference complete. Output length: {len(output_text)}")
    # Basic cleanup - sometimes models add the prompt again or extra spaces
    clean_output = output_text.strip()
    # Optional: more aggressive cleanup if needed, e.g., removing prompt remnants
    # if clean_output.startswith(prompt[-50:]): # Heuristic check
    #     clean_output = clean_output[len(prompt[-50:]):].strip()

    # Often, the model might just output the code block. Try to extract it if fenced.
    code_match = re.search(r"```(?:mal)?\s*(.*?)\s*```", clean_output, re.DOTALL | re.IGNORECASE)
    if code_match:
        logging.info("Extracted MAL code from fenced block.")
        return code_match.group(1).strip()
    else:
        # If no fence, assume the whole output is the code (might need refinement)
        # Or potentially just return the clean_output and let validation handle it
        logging.info("No fenced code block found, returning raw cleaned output.")
        return clean_output # Return the cleaned output directly


def run_mal_compiler(code_str: str, compiler_executable_path: str, malc_args: list = None) -> tuple[bool, str]:
    """
    Compiles MAL code using the specified malc executable binary.

    Args:
        code_str: The MAL code as a string.
        compiler_executable_path: Path to the malc executable binary.
        malc_args: Optional list of additional arguments for malc.

    Returns:
        A tuple (success: bool, message: str), where message is stdout on success,
        or stderr/error message on failure.
    """
    if malc_args is None:
        malc_args = []

    temp_file_path = None # Initialize path variable
    try:
        # Create a temporary file to hold the code
        # delete=False needed because we pass the path to subprocess
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mal', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(code_str)
            temp_file_path = tmp_file.name # Get the path before closing

        # Ensure the MALC executable path exists and is executable
        if not os.path.exists(compiler_executable_path):
            logging.error(f"malc executable not found at path: {compiler_executable_path}")
            return False, f"Compiler Error: Executable not found at {compiler_executable_path}"
        if not os.access(compiler_executable_path, os.X_OK):
             logging.warning(f"malc executable at {compiler_executable_path} may not be executable. Attempting to set +x.")
             try:
                 os.chmod(compiler_executable_path, os.stat(compiler_executable_path).st_mode | 0o111) # Add execute permissions
                 if not os.access(compiler_executable_path, os.X_OK):
                     raise OSError("Failed to set executable permission.")
             except OSError as chmod_err:
                 logging.error(f"Failed to make malc executable: {chmod_err}")
                 return False, f"Compiler Error: Could not ensure executable permission for {compiler_executable_path}"


        # Build the command list
        command = [compiler_executable_path]
        command.append(temp_file_path) # Add the path to the temp MAL file
        command.extend(malc_args)      # Add any extra arguments

        # Log the command being executed (useful for debugging)
        logging.info(f"Executing malc command: {' '.join(shlex.quote(str(arg)) for arg in command)}") # Ensure all args are strings

        # Run the compiler process
        process = subprocess.run(
            command,
            capture_output=True, # Captures stdout and stderr
            text=True,           # Decode output as text using default encoding
            timeout=30,          # Timeout in seconds
            check=False          # Do not raise exception on non-zero exit code
        )

        # Check the result
        if process.returncode == 0:
            logging.info(f"malc compilation successful. Stdout (first 500 chars):\n{process.stdout[:500]}")
            return True, process.stdout
        else:
            # Combine stdout and stderr for error message as context might be in either
            error_message = f"Stderr:\n{process.stderr}\nStdout:\n{process.stdout}"
            logging.warning(f"malc compilation failed. Return code: {process.returncode}. Combined Output:\n{error_message}")
            return False, error_message.strip() # Return combined output as error

    except subprocess.TimeoutExpired:
        logging.error(f"malc compilation timed out after 30 seconds.")
        return False, "Compiler error: Process timed out."
    # FileNotFoundError is implicitly checked by os.path.exists now
    except Exception as e:
        logging.error(f"An unexpected error occurred during malc execution: {e}", exc_info=True)
        return False, f"Compiler error: An unexpected Python error occurred: {str(e)}"
    finally:
        # Ensure the temporary file is deleted even if errors occur
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                # logging.debug(f"Deleted temporary MAL file: {temp_file_path}")
            except OSError as e:
                logging.error(f"Error deleting temporary file {temp_file_path}: {e}")

def load_llm_and_tokenizer(base_model_id, adapter_path=None):
    """Loads the LLM and tokenizer, optionally merging an adapter."""
    try:
        logging.info(f"Loading base model: {base_model_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True, # Recommended practice
            bnb_4bit_quant_type="nf4",      # Recommended practice
            bnb_4bit_compute_dtype=torch.bfloat16 # Recommended practice
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto", # Automatically distribute across GPUs if available
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        logging.info("Base model loaded.")

        if adapter_path and os.path.exists(adapter_path):
            logging.info(f"Loading and merging adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload() # Merge adapter into the base model
            logging.info("Adapter merged successfully.")
        elif adapter_path:
            logging.warning(f"Adapter path specified ({adapter_path}), but not found. Using base model only.")
        else:
            logging.info("No adapter path specified. Using base model only.")

        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        # Set padding token if it's not already set (common issue with some models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Set tokenizer pad_token to eos_token.")

        return model, tokenizer
    except Exception as e:
        logging.error(f"Model/tokenizer loading error: {e}", exc_info=True)
        return None, None

def generate_valid_mal(input_text, model, tokenizer, embedding_model, faiss_index, docstore_map):
    """Generates MAL code, attempting refinement based on compiler feedback."""
    logging.info(f"Starting MAL code generation for input: '{input_text[:100]}...'")
    current_code = ""
    compiler_error = ""
    rag_context_string = ""

    # Retrieve context once at the beginning
    if USE_FAISS and embedding_model and faiss_index and docstore_map:
         retrieved_docs_content = get_rag_examples(
             input_text, embedding_model, faiss_index, docstore_map, TOP_K_RAG
         )
         rag_context_string = "\n\n---\n\n".join(retrieved_docs_content)
         if not rag_context_string:
             rag_context_string = "No relevant context found in CAPEC data."
             logging.info("RAG retrieved no documents.")
         else:
             logging.info("RAG context retrieved.")
    else:
        rag_context_string = "RAG context is not available or disabled."
        logging.info("RAG context skipped.")


    for attempt in range(MAX_ATTEMPTS):
        logging.info(f"Generation attempt {attempt + 1}/{MAX_ATTEMPTS}")

        if attempt == 0:
            prompt = INITIAL_PROMPT_TEMPLATE.format(
                context=rag_context_string,
                input_text=input_text
            )
        else:
            # Use refinement prompt if previous attempt failed
            prompt = REFINEMENT_PROMPT_TEMPLATE.format(
                compiler_error=compiler_error,
                context=rag_context_string, # Provide context again for refinement
                input_text=input_text,      # Provide original request again
                previous_code=current_code  # Provide the faulty code
            )

        # Generate code using the LLM
        llm_output = run_llm_inference(prompt, model, tokenizer)

        # Basic check if LLM output seems empty or placeholder
        if not llm_output or len(llm_output) < 10: # Arbitrary short length check
             logging.warning(f"LLM generated very short or empty output on attempt {attempt + 1}. Output: '{llm_output}'")
             compiler_error = "LLM generated empty or insufficient code."
             current_code = llm_output # Store it anyway for the next prompt
             continue # Go to next attempt

        current_code = llm_output # Update current code with the new generation

        # Validate the generated code using the MAL compiler
        # Pass the configured path to the malc executable here
        success, message = run_mal_compiler(current_code, MALC_EXECUTABLE_PATH)

        if success:
            logging.info("MAL code compiled successfully!")
            return current_code # Return the valid code
        else:
            logging.warning(f"Compilation failed on attempt {attempt + 1}.")
            compiler_error = message # Store the error message for the next refinement prompt

    # If loop finishes without success
    logging.error(f"Failed to generate valid MAL code after {MAX_ATTEMPTS} attempts.")
    return f"Error: Could not generate valid MAL code after {MAX_ATTEMPTS} attempts.\nLast Code Attempt:\n{current_code}\nLast compiler error:\n{compiler_error}"

if __name__ == "__main__":
    logging.info("--- Initializing MAL Agent ---")

    # Load Model and Tokenizer
    model, tokenizer = load_llm_and_tokenizer(BASE_MODEL_ID, ADAPTER_PATH)
    if model is None or tokenizer is None:
        logging.critical("FATAL: Model or tokenizer loading failed. Exiting.")
        exit(1)

    # Setup RAG components
    faiss_index, docstore_map, embedding_model = setup_rag(EMBEDDING_MODEL_ID, VECTOR_STORE_PATH)
    if USE_FAISS and (faiss_index is None or docstore_map is None or embedding_model is None):
        # Log as critical only if RAG was intended but failed
        logging.critical(f"FATAL: RAG setup failed (index/docstore path: {VECTOR_STORE_PATH}). Exiting as RAG is required but failed.")
        exit(1)
    elif not USE_FAISS:
        logging.info("RAG is disabled by configuration (USE_FAISS=False).")


    # --- Example Usage ---
    # TODO: Replace with your actual input mechanism
    example_input = "Describe an attack where an adversary uses SQL injection to gain initial access and then escalates privileges via a known kernel exploit, targeting a web server which connects to a database."

    # Generate the MAL code
    final_mal_code = generate_valid_mal(
        example_input,
        model,
        tokenizer,
        embedding_model, # Will be None if RAG is disabled/failed
        faiss_index,     # Will be None if RAG is disabled/failed
        docstore_map     # Will be None if RAG is disabled/failed
    )

    print("\n" + "="*30 + " Final MAL Code " + "="*30)
    print(final_mal_code)
    print("="*76)
    logging.info("--- MAL Agent Finished ---")