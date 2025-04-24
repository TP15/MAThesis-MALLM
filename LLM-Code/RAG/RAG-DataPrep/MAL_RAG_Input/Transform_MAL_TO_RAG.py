import json
import os

def transform_mal_entry(mal_data):
    """
    Transforms a single MAL entry (from JSONL) into the target RAG format.

    Args:
        mal_data (dict): A dictionary representing a single line from the MAL JSONL file.
                         Expected keys: "input", "output", "type".

    Returns:
        dict or None: The transformed data in the target format, or None if input is invalid.
    """
    if not isinstance(mal_data, dict):
        print(f"Skipping invalid MAL data (expected dict): {type(mal_data)}")
        return None

    # Use .get() to safely access keys, providing default values if they might be missing
    description = mal_data.get("input", "") # Natural language description
    mal_code = mal_data.get("output", "")  # The actual MAL code
    mal_type = mal_data.get("type")      # e.g., "asset"

    # Combine description and code for embedding. Add separators for clarity.
    # You can adjust this combination based on what works best for your RAG retrieval.
    embedding_input = f"Description:\n{description}\n\nMAL Code:\n{mal_code}"

    # Define metadata - primarily the type from the source
    metadata = {}
    if mal_type is not None:
        metadata["mal_type"] = mal_type
    # You could add more metadata here if needed, e.g., by parsing the mal_code,
    # but that would require a MAL parser.

    # Construct the final object
    return {
        "embedding_input": embedding_input,
        "source_type": "MAL", # Set source type specifically to MAL
        "metadata": metadata,
        "raw": mal_data # Store the original MAL entry
    }

def process_mal_jsonl(mal_input_file, mal_output_file):
    """
    Reads a MAL JSONL file, transforms each entry, and writes the results
    to a new JSONL file in the target RAG format.

    Args:
        mal_input_file (str): Path to the input MAL JSONL file.
        mal_output_file (str): Path for the output JSONL file.
    """
    processed_count = 0
    error_count = 0

    # Ensure output directory exists
    output_dir = os.path.dirname(mal_output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Processing MAL data from: {mal_input_file}")
    print(f"Writing transformed data to: {mal_output_file}")

    try:
        with open(mal_input_file, 'r', encoding='utf-8') as infile, \
             open(mal_output_file, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                try:
                    original_data = json.loads(line)
                    transformed_data = transform_mal_entry(original_data)

                    if transformed_data:
                        json.dump(transformed_data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        processed_count += 1
                    else:
                        # Error details should have been printed by transform_mal_entry
                        error_count += 1

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line #{line_num} in {mal_input_file}: {line[:100]}...")
                    error_count += 1
                except Exception as e:
                    print(f"Error processing line #{line_num} in {mal_input_file}: {line[:100]}... Error: {e}")
                    error_count += 1

    except FileNotFoundError:
        print(f"Error: Input file not found at {mal_input_file}")
        error_count += 1 # Consider file not found as an error
    except IOError as e:
        print(f"Error accessing files. Input: {mal_input_file}, Output: {mal_output_file}. Error: {e}")
        # No point reporting counts if files couldn't be opened/written
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # No point reporting counts if a major error occurred
        return


    print(f"\nTransformation complete.")
    print(f"Successfully processed and wrote {processed_count} MAL entries.")
    if error_count > 0:
        print(f"Encountered {error_count} errors or skipped entries.")

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy input MAL JSONL file for the example
    if not os.path.exists("temp_input"):
        os.makedirs("temp_input")

    dummy_mal_input_file = os.path.join("temp_input", "mal_data.jsonl")
    mal_rag_output_file = "output/mal_rag_data.jsonl" # Specify your desired output path

    # Example MAL entry (ensure it's written as one line in the file)
    mal_entry_1 = {
        "input": "This MAL code defines a series of attack steps...", # Truncated for brevity
        "output": "asset Hardware\n      user info: \"Specifies the hardware...\"...", # Truncated for brevity
        "type": "asset"
    }
    mal_entry_2 = { # Another example
        "input": "Describes a network connection asset.",
        "output": "asset Network\n    {\n      | connect\n    }",
        "type": "asset"
    }

    # Write dummy data to the JSONL file (one JSON object per line)
    try:
        with open(dummy_mal_input_file, 'w', encoding='utf-8') as f:
            json.dump(mal_entry_1, f, ensure_ascii=False)
            f.write('\n')
            json.dump(mal_entry_2, f, ensure_ascii=False)
            f.write('\n')
            f.write('{"bad json"\n') # Add an invalid line for testing errors
            f.write('{"input": "Only input key"}\n') # Add incomplete line

        print("--- Starting MAL Transformation ---")
        # *** IMPORTANT: Replace dummy paths with your actual file paths below ***
        process_mal_jsonl(
            mal_input_file="/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/MAL_RAG.jsonl",    # <- Your MAL input file path
            mal_output_file="/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/Transformed_MALRAG_Data.jsonl"     # <- Your desired output file path
        )
        print("--- MAL Transformation Finished ---")

        # Optional: Preview the output file
        if os.path.exists(mal_rag_output_file):
            print(f"\nFirst few lines of {mal_rag_output_file}:")
            try:
                with open(mal_rag_output_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 3: # Show max 3 lines
                            break
                        print(line.strip())
            except Exception as e:
                 print(f"Could not read output file for preview: {e}")

    except Exception as e:
        print(f"Error setting up or running example: {e}")