import json
import os

def extract_capec_id(external_references):
    """
    Sucht in der Liste der externen Referenzen nach dem CAPEC-Eintrag
    und gibt die externe ID zurück.
    """
    if not external_references:
        return None
    for ref in external_references:
        if ref.get("source_name") == "capec" and "external_id" in ref:
            return ref["external_id"]
    return None

def transform_attack_pattern(ap_data):
    """
    Transformiert einen Attack Pattern Eintrag in das Zielformat.
    """
    if not isinstance(ap_data, dict):
        print(f"Skipping invalid attack pattern data: {ap_data}")
        return None

    embedding_input = f"Name: {ap_data.get('name', 'N/A')}\nDescription: {ap_data.get('description', 'N/A')}"

    metadata = {
        "id": ap_data.get("id"),
        "type": ap_data.get("type"),
        "name": ap_data.get("name"),
        "capec_id": extract_capec_id(ap_data.get("external_references")),
        "abstraction": ap_data.get("x_capec_abstraction"),
        "domains": ap_data.get("x_capec_domains"),
        "status": ap_data.get("x_capec_status"),
        "version": ap_data.get("x_capec_version")
    }
    # Entferne Metadaten-Felder, die None sind, um das Objekt sauber zu halten
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return {
        "embedding_input": embedding_input,
        "source_type": "CAPEC", # Wie besprochen, Quelle ist CAPEC
        "metadata": metadata,
        "raw": ap_data # Der komplette Originaleintrag
    }

def transform_course_of_action(coa_data):
    """
    Transformiert einen Course of Action Eintrag in das Zielformat.
    """
    if not isinstance(coa_data, dict):
        print(f"Skipping invalid course of action data: {coa_data}")
        return None

    embedding_input = f"Name: {coa_data.get('name', 'N/A')}\nDescription: {coa_data.get('description', 'N/A')}"

    metadata = {
        "id": coa_data.get("id"),
        "type": coa_data.get("type"),
        "name": coa_data.get("name"),
        "version": coa_data.get("x_capec_version")
    }
    # Entferne Metadaten-Felder, die None sind
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return {
        "embedding_input": embedding_input,
        "source_type": "CAPEC", # Quelle ist ebenfalls CAPEC-Kontext
        "metadata": metadata,
        "raw": coa_data # Der komplette Originaleintrag
    }

# --- MODIFIED FUNCTION ---
def transform_and_combine_jsonl(attack_pattern_file, course_of_action_file, output_file):
    """
    Liest Attack Pattern und Course of Action JSON-Dateien (die jeweils
    eine Liste von Objekten enthalten), transformiert jeden Eintrag und
    schreibt sie in eine gemeinsame Output-JSONL-Datei.

    Args:
        attack_pattern_file (str): Pfad zur Attack Pattern JSON-Datei (Liste).
        course_of_action_file (str): Pfad zur Course of Action JSON-Datei (Liste).
        output_file (str): Pfad zur zu erstellenden Output-JSONL-Datei.
    """
    processed_count = 0
    error_count = 0

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:

            # --- Process Attack Patterns ---
            print(f"Processing Attack Patterns from: {attack_pattern_file}")
            try:
                with open(attack_pattern_file, 'r', encoding='utf-8') as infile:
                    content = infile.read() # Read the whole file
                    try:
                        # Parse the entire content as a JSON list
                        data_list = json.loads(content)
                        if not isinstance(data_list, list):
                            print(f"Error: Expected a JSON list in {attack_pattern_file}, but got {type(data_list)}")
                            error_count += 1 # Count the whole file as an error
                        else:
                             # Iterate through items in the list
                            for original_data in data_list:
                                try:
                                    transformed_data = transform_attack_pattern(original_data)
                                    if transformed_data:
                                        json.dump(transformed_data, outfile, ensure_ascii=False)
                                        outfile.write('\n') # Write as JSON Lines
                                        processed_count += 1
                                    else:
                                        # Error already printed in transform function if data was invalid type
                                        error_count += 1
                                except Exception as e:
                                    print(f"Error processing attack pattern item: {original_data.get('id', 'N/A')}. Error: {e}")
                                    error_count += 1

                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON file: {attack_pattern_file}. Error: {e}")
                        error_count += 1 # Count the whole file as an error
            except FileNotFoundError:
                print(f"Error: Attack Pattern file not found at {attack_pattern_file}")
                error_count += 1
            except Exception as e:
                 print(f"An unexpected error occurred while processing {attack_pattern_file}: {e}")
                 error_count += 1


            # --- Process Courses of Action ---
            print(f"\nProcessing Courses of Action from: {course_of_action_file}")
            try:
                with open(course_of_action_file, 'r', encoding='utf-8') as infile:
                    content = infile.read() # Read the whole file
                    try:
                        # Parse the entire content as a JSON list
                        data_list = json.loads(content)
                        if not isinstance(data_list, list):
                           print(f"Error: Expected a JSON list in {course_of_action_file}, but got {type(data_list)}")
                           error_count += 1
                        else:
                            # Iterate through items in the list
                            for original_data in data_list:
                                try:
                                    transformed_data = transform_course_of_action(original_data)
                                    if transformed_data:
                                        json.dump(transformed_data, outfile, ensure_ascii=False)
                                        outfile.write('\n') # Write as JSON Lines
                                        processed_count += 1
                                    else:
                                        # Error already printed in transform function if data was invalid type
                                        error_count += 1
                                except Exception as e:
                                    print(f"Error processing course of action item: {original_data.get('id', 'N/A')}. Error: {e}")
                                    error_count += 1

                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON file: {course_of_action_file}. Error: {e}")
                        error_count += 1
            except FileNotFoundError:
                print(f"Error: Course of Action file not found at {course_of_action_file}")
                error_count += 1
            except Exception as e:
                 print(f"An unexpected error occurred while processing {course_of_action_file}: {e}")
                 error_count += 1

    except IOError as e:
        print(f"Error opening or writing to output file {output_file}: {e}")
        return

    print(f"\nTransformation complete.")
    print(f"Successfully processed and wrote {processed_count} entries to {output_file}")
    if error_count > 0:
        print(f"Encountered {error_count} errors or skipped entries/files.")


# --- MODIFIED Example Usage ---
if __name__ == "__main__":
    print("--- Starting Transformation ---")
    transform_and_combine_jsonl(
        attack_pattern_file="/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/capec_rag_input_data/attack-pattern_rag_data.json", # <- ÄNDERN (Pfad zu deiner AP JSON-Datei)
        course_of_action_file="/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/capec_rag_input_data/course-of-action_rag_data.json", # <- ÄNDERN (Pfad zu deiner CoA JSON-Datei)
        output_file="/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/capec_rag_input_data/capec_combined_rag_data.jsonl"    # <- ÄNDERN (Zielpfad für die JSONL-Datei)
    )
    print("--- Transformation Finished ---")