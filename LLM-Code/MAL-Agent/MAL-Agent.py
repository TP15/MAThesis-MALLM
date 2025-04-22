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
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "./path/to/your/qlora_adapter"
MAL_COMPILER_PATH = "/usr/local/bin/mal-compiler"

# --- RAG Configuration ---
EMBEDDING_MODEL_ID = 'all-MiniLM-L6-v2'
VECTOR_STORE_PATH = "capec_faiss_index"
USE_FAISS = True
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
    logging.info(f"Setting up RAG by loading index from: {index_folder_path}")
    faiss_index_file = os.path.join(index_folder_path, "index.faiss")
    docstore_file = os.path.join(index_folder_path, "index.pkl")

    if not os.path.exists(faiss_index_file) or not os.path.exists(docstore_file):
        logging.error(f"FAISS index ('index.faiss') or docstore ('index.pkl') not found in {index_folder_path}")
        return None, None, None

    try:
        faiss_index = faiss.read_index(faiss_index_file)

        with open(docstore_file, "rb") as f:
            docstore = pickle.load(f)

        if isinstance(docstore, tuple) and len(docstore) == 2:
            index_to_id = docstore[0]
            actual_docstore = docstore[1]
            docstore_map = {i: actual_docstore[index_to_id[str(i)]] for i in range(faiss_index.ntotal)}
        elif isinstance(docstore, dict):
            docstore_map = docstore
        else:
            logging.error("Unexpected format loaded from index.pkl.")
            return None, None, None

        embedding_model = SentenceTransformer(embedding_model_id)
        return faiss_index, docstore_map, embedding_model

    except Exception as e:
        logging.error(f"RAG setup error: {e}", exc_info=True)
        return None, None, None

def get_rag_examples(query_text, embedding_model, faiss_index, docstore_map, top_k):
    if not faiss_index or not docstore_map or not embedding_model:
        logging.warning("RAG components not available.")
        return []

    try:
        query_embedding = embedding_model.encode([query_text], normalize_embeddings=True)
        distances, indices = faiss_index.search(query_embedding, top_k)

        retrieved_contents = []
        for i in indices[0]:
            if i != -1 and i in docstore_map:
                document = docstore_map[i]
                retrieved_contents.append(document.page_content)
        return retrieved_contents
    except Exception as e:
        logging.error(f"RAG retrieval error: {e}", exc_info=True)
        return []

def run_llm_inference(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_mal_compiler(code_str, compiler_path):
    try:
        with tempfile.NamedTemporaryFile("w+", suffix=".mal", delete=False) as tmp_file:
            tmp_file.write(code_str)
            tmp_file.flush()
            result = subprocess.run(
                [compiler_path, tmp_file.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
    except Exception as e:
        return False, f"Compiler error: {str(e)}"

def load_llm_and_tokenizer(base_model_id, adapter_path):
    try:
        logging.info("Loading base model...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model loading error: {e}", exc_info=True)
        return None, None

def generate_valid_mal(input_text, model, tokenizer, embedding_model, faiss_index, docstore_map):
    logging.info("Generating MAL code...")
    current_code = ""
    compiler_error = ""
    rag_context_string = ""

    for attempt in range(MAX_ATTEMPTS):
        logging.info(f"Attempt {attempt + 1}/{MAX_ATTEMPTS}")

        if attempt == 0:
            retrieved_docs_content = get_rag_examples(
                input_text, embedding_model, faiss_index, docstore_map, TOP_K_RAG
            )
            rag_context_string = "\n\n---\n\n".join(retrieved_docs_content)
            if not rag_context_string:
                rag_context_string = "No relevant context found in CAPEC data."

        prompt = (
            INITIAL_PROMPT_TEMPLATE if attempt == 0 else REFINEMENT_PROMPT_TEMPLATE
        ).format(
            context=rag_context_string,
            input_text=input_text,
            previous_code=current_code,
            compiler_error=compiler_error
        )

        current_code = run_llm_inference(prompt, model, tokenizer)
        success, message = run_mal_compiler(current_code, MAL_COMPILER_PATH)

        if success:
            logging.info("MAL code compiled successfully.")
            return current_code
        else:
            logging.warning("Compilation failed.")
            compiler_error = message

    logging.error("Failed to generate valid MAL code.")
    return f"Error: Could not generate valid MAL code.\nLast compiler error:\n{compiler_error}"

if __name__ == "__main__":
    logging.info("--- Initializing MAL Agent ---")

    model, tokenizer = load_llm_and_tokenizer(BASE_MODEL_ID, ADAPTER_PATH)
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer loading failed.")
        exit(1)

    faiss_index, docstore_map, embedding_model = setup_rag(EMBEDDING_MODEL_ID, VECTOR_STORE_PATH)
    if faiss_index is None or docstore_map is None or embedding_model is None:
        logging.error("RAG setup failed.")
        exit(1)

    example_input = "Describe an attack where an adversary uses SQL injection to gain initial access and then escalates privileges via a known kernel exploit."
    final_mal_code = generate_valid_mal(
        example_input,
        model,
        tokenizer,
        embedding_model,
        faiss_index,
        docstore_map
    )

    print("\n--- Final MAL Code ---")
    print(final_mal_code)
    print("--- Finished ---")