import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import os # Imported for file path checking

# --- Basic Configuration ---
BASE_MODEL_ID = "mistralai/Mistral-7B-"  # Example, adapt to your model
ADAPTER_PATH = "/content/Instruct"  # Example, adapt to your adapter path
                                     # or None if you are not using an adapter
PROMPT_FILE_PATHS = [
    "prompts_general.txt",
    "prompts_technical.txt",
    "prompts_creative.txt"
]  # List of file names from which prompts will be loaded

# --- LLM Inference Configuration ---
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

def load_llm_and_tokenizer(base_model_id, adapter_path=None):
    """
    Loads the base LLM and optionally a PEFT adapter, along with the tokenizer.
    """
    print(f"Loading base model: {base_model_id}...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("Base model loaded.")

        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading PEFT adapter from: {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
            print("PEFT adapter loaded.")
        elif adapter_path:
            print(f"Warning: Adapter path '{adapter_path}' does not exist. Proceeding without adapter.")
            print("No PEFT adapter will be loaded.")
        else:
            print("No PEFT adapter will be loaded.")


        print(f"Loading tokenizer for: {base_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Tokenizer pad_token has been set to eos_token.")
        print("Tokenizer loaded.")

        print("LLM and Tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def run_llm_inference(prompt, model, tokenizer):
    """
    Performs inference with the LLM and returns the response.
    """
    cleaned_prompt_display = prompt[:100].replace(os.linesep, ' ')
    print(f"\nSending prompt to LLM (Length: {len(prompt)} characters): \"{cleaned_prompt_display}...\"")
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if decoded_output.startswith(prompt):
            final_output = decoded_output[len(prompt):].strip()
        else:
            final_output = decoded_output.strip()

        print("LLM inference complete.")
        return final_output
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        return "Error during inference."

def load_prompts_from_files(file_paths):
    """
    Loads prompts from a list of text files.
    Each line in each file is treated as one prompt.
    Empty lines are ignored.
    """
    all_prompts = []
    successful_files_count = 0
    failed_files_count = 0

    if not file_paths:
        print("Warning: No prompt file paths were provided.")
        return all_prompts

    print(f"Attempting to load prompts from {len(file_paths)} specified file(s)...")
    for file_path in file_paths:
        prompts_from_current_file = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                prompts_from_current_file = [line.strip() for line in f if line.strip()]
            
            if not prompts_from_current_file:
                print(f"Info: File '{file_path}' is empty or contains only empty lines. No prompts loaded from this file.")
            else:
                all_prompts.extend(prompts_from_current_file)
                print(f"Successfully loaded {len(prompts_from_current_file)} prompts from '{file_path}'.")
                successful_files_count +=1
        except FileNotFoundError:
            print(f"Error: Prompt file '{file_path}' not found.")
            failed_files_count += 1
        except Exception as e:
            print(f"An error occurred while reading file '{file_path}': {e}")
            failed_files_count += 1
    
    print(f"Finished loading prompts. Total prompts loaded: {len(all_prompts)}. "
          f"Successfully processed files: {successful_files_count}. Failed files: {failed_files_count}.")
    return all_prompts

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting simple LLM prompting script ---")

    # Optional: Hugging Face Login, if required for your model
    # Attempt to log in, but continue if it fails or no token is provided
    try:
        hf_token = None # Set your token here directly or load it from an environment variable
                        # Example: hf_token = "hf_YOUR_TOKEN_HERE"
        if hf_token:
            login(token=hf_token)
            print("Successfully logged into Hugging Face.")
        else:
            print("No Hugging Face token found/provided for login. Proceeding without.")
    except Exception as e:
        print(f"Error during Hugging Face login (optional): {e}")

    # Load model and tokenizer
    # Use ADAPTER_PATH=None if you do not want to load an adapter.
    model, tokenizer = load_llm_and_tokenizer(BASE_MODEL_ID, ADAPTER_PATH)

    if model is None or tokenizer is None:
        print("Model or tokenizer could not be loaded. Exiting script.")
        exit(1)

    # Load prompts from the specified files
    prompts_to_send = load_prompts_from_files(PROMPT_FILE_PATHS)

    if not prompts_to_send:
        print("No prompts to send. Please check the specified prompt files.")
        print("Ensure the files listed in PROMPT_FILE_PATHS exist, are in the same directory as the script (or provide full paths), and contain prompts (one per line).")
        print("Exiting script.")
        exit(1)

    print(f"\n--- Starting to send {len(prompts_to_send)} prompts ---")
    for i, user_prompt in enumerate(prompts_to_send):
        print(f"\n--- Processing Prompt {i+1}/{len(prompts_to_send)} ---")
        llm_response = run_llm_inference(user_prompt, model, tokenizer)
        print(f"\nResponse from LLM for Prompt {i+1}:")
        print("--------------------------------------------------")
        print(llm_response)
        print("--------------------------------------------------")

    print("\n--- All prompts have been processed. Exiting script. ---")