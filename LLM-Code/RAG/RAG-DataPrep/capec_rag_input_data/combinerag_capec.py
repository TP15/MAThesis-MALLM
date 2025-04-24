import os
import sys

def combine_jsonl(file1_path, file2_path, output_file_path):
    """
    Combines two JSONL files into a single JSONL file.

    Args:
        file1_path (str): Path to the first input JSONL file.
        file2_path (str): Path to the second input JSONL file.
        output_file_path (str): Path for the combined output JSONL file.
    """
    # Basic check to prevent overwriting input files
    if output_file_path == file1_path or output_file_path == file2_path:
        print(f"Error: Output file path cannot be the same as an input file path.")
        sys.exit(1) # Exit with an error code

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            sys.exit(1)

    print(f"Combining '{os.path.basename(file1_path)}' and '{os.path.basename(file2_path)}' into '{os.path.basename(output_file_path)}'...")

    total_lines_written = 0
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # Process first file
            try:
                print(f"Processing '{file1_path}'...")
                lines_file1 = 0
                with open(file1_path, 'r', encoding='utf-8') as infile1:
                    for line in infile1:
                        outfile.write(line) # Write line directly (includes newline)
                        lines_file1 += 1
                total_lines_written += lines_file1
                print(f"  Added {lines_file1} lines from '{os.path.basename(file1_path)}'.")
            except FileNotFoundError:
                print(f"Error: Input file not found: {file1_path}")
                # Decide if you want to continue or exit if a file is missing
                # sys.exit(1)
                # Or just print warning and continue with the next file

            # Process second file
            try:
                print(f"Processing '{file2_path}'...")
                lines_file2 = 0
                with open(file2_path, 'r', encoding='utf-8') as infile2:
                    for line in infile2:
                        outfile.write(line) # Write line directly
                        lines_file2 += 1
                total_lines_written += lines_file2
                print(f"  Added {lines_file2} lines from '{os.path.basename(file2_path)}'.")
            except FileNotFoundError:
                 print(f"Error: Input file not found: {file2_path}")
                 # Decide if you want to continue or exit

        print(f"\nSuccessfully combined files.")
        print(f"Total lines written to '{output_file_path}': {total_lines_written}")

    except IOError as e:
        print(f"Error during file operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Starting Combination ---")

    combine_jsonl(
        file1_path="/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/MAL_RAG_Input/Transformed_MALRAG_Data.jsonl",      # <- Your first input JSONL file
        file2_path="/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/capec_rag_input_data/capec_combined_rag_data.jsonl",      # <- Your second input JSONL file
        output_file_path="/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/final_RAG_MAL_CAPEC_DATA.jsonl" # 
    )
    print("--- Combination Finished ---")