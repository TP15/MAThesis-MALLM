import os
import json # <--- Import json module
# We removed requests, shutil as we rely on the local clone now
from stix2 import FileSystemSource, Filter
from typing import Dict, List, Set, Any, Optional

# --- Configuration ---

# IMPORTANT: SET THIS PATH to the location where you cloned the mitre/cti repository
# Example: '/path/to/your/projects/cti' or 'C:/Users/YourUser/Documents/cti'
LOCAL_CTI_REPO_PATH = '/Users/thomaspathe/Documents/MAThesis-MALLM/CTI/cti' # <--- CHANGE THIS

# Directory where the processed RAG input files will be saved
OUTPUT_RAG_DIR = "capec_rag_input_data" # <--- Name of the output folder

# Define the STIX object types you are interested in extracting from CAPEC
DESIRED_CAPEC_TYPES: Set[str] = {
    "attack-pattern",    # CAPEC Attack Patterns
    "course-of-action",  # CAPEC Mitigations
}
# --- End Configuration ---


def get_capec_id(stix_object: Dict[str, Any]) -> Optional[str]:
    """
    Extracts the CAPEC ID from a STIX object's external_references.

    Args:
        stix_object: A STIX object (as a dictionary or stix2 object).

    Returns:
        The CAPEC ID (e.g., "CAPEC-66") or None if not found.
    """
    if not hasattr(stix_object, 'external_references'):
        return None

    for ref in stix_object.external_references:
        if ref.get('source_name') == 'capec' and ref.get('external_id'):
            ext_id = ref['external_id']
            if isinstance(ext_id, int):
                 return f"CAPEC-{ext_id}"
            elif isinstance(ext_id, str):
                 return ext_id if ext_id.startswith("CAPEC-") else f"CAPEC-{ext_id}"
    return None

# --- NEW FUNCTION TO SAVE DATA ---
def save_for_rag(data_dict: Dict[str, List[Any]], output_dir: str):
    """
    Saves the extracted STIX objects into JSON files suitable for RAG input.

    Each object type gets its own JSON file containing a list of objects,
    where each object is converted to a standard Python dictionary.

    Args:
        data_dict: The dictionary containing lists of stix2 objects per type
                   (e.g., {'attack-pattern': [obj1, obj2], ...}).
        output_dir: The path to the directory where JSON files will be saved.
    """
    print(f"\nSaving data for RAG input into directory: {output_dir}")
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists or created it.")
    except OSError as e:
        print(f"  Error creating directory {output_dir}: {e}")
        return # Stop if directory cannot be created

    for obj_type, object_list in data_dict.items():
        if not object_list:
            print(f"  Skipping type '{obj_type}': No objects found.")
            continue

        # Convert the list of stix2 objects to a list of dictionaries
        # using the .serialize() method, which gives a JSON-compatible string,
        # then parse it back to a dict. This handles custom properties correctly.
        data_to_save = []
        print(f"  Processing {len(object_list)} objects of type '{obj_type}' for saving...")
        for stix_obj in object_list:
             try:
                 # serialize() gives a string, json.loads() makes it a dict
                 obj_dict = json.loads(stix_obj.serialize())
                 data_to_save.append(obj_dict)
             except Exception as e:
                 print(f"    Warning: Could not serialize object {getattr(stix_obj, 'id', 'N/A')}: {e}")


        if not data_to_save:
             print(f"  Skipping file for '{obj_type}': No objects could be serialized.")
             continue

        # Define the output filename
        file_name = f"{obj_type}_rag_data.json"
        file_path = os.path.join(output_dir, file_name)

        print(f"  Saving {len(data_to_save)} '{obj_type}' objects to {file_path}...")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Use json.dump for writing the list of dicts to the file
                # indent=4 makes the file readable
                # ensure_ascii=False handles special characters correctly
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            print(f"    Successfully saved.")
        except IOError as e:
            print(f"    Error saving file {file_path}: {e}")
        except TypeError as e:
             print(f"    Error during JSON serialization for {file_path}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Construct the path to the capec data within the cloned repository
    capec_data_path = os.path.join(LOCAL_CTI_REPO_PATH, 'capec', '2.1')

    print(f"Attempting to access CAPEC data in: {capec_data_path}")

    # --- Pre-check: Verify the path exists ---
    if not os.path.isdir(capec_data_path):
         print("\n--- ERROR ---")
         print(f"The specific STIX version directory was not found: {capec_data_path}")
         print(f"Please ensure the repository at '{LOCAL_CTI_REPO_PATH}' is complete and contains the 'capec/2.1/' structure.")
         print("-------------\n")
         exit(1)
    # --- End Pre-check ---

    fs = None
    capec_data: Dict[str, List[Any]] = {obj_type: [] for obj_type in DESIRED_CAPEC_TYPES}

    try:
        # 1. Initialize the stix2 FileSystemSource
        print(f"\nInitializing STIX FileSystemSource for directory: {capec_data_path}")
        fs = FileSystemSource(capec_data_path, allow_custom=True)
        print("FileSystemSource initialized successfully.")

        # 2. Query for the desired object types
        print("\nQuerying for desired object types...")
        for obj_type in DESIRED_CAPEC_TYPES:
            try:
                filt = Filter('type', '=', obj_type)
                objects = fs.query([filt])
                capec_data[obj_type] = objects
                print(f"  Found {len(objects)} objects of type '{obj_type}'")
            except Exception as e:
                 print(f"  Error querying for type '{obj_type}': {e}")

        # 3. Example: Access data from the first Attack Pattern (Optional display)
        if capec_data.get("attack-pattern"):
            print("\n--- Example: First CAPEC Attack Pattern ---")
            # ... (example display code remains the same) ...
            first_ap = capec_data["attack-pattern"][0]
            capec_id = get_capec_id(first_ap)
            print(f"  STIX ID: {first_ap.id}")
            print(f"  CAPEC ID: {capec_id or 'Not Found'}")
            print(f"  Name: {getattr(first_ap, 'name', 'N/A')}")
            print(f"  Description: {getattr(first_ap, 'description', 'N/A')[:150]}...")
            print(f"  Custom Abstraction: {getattr(first_ap, 'x_capec_abstraction', 'N/A')}")
            prereqs = getattr(first_ap, 'x_capec_prerequisites', [])
            print(f"  Custom Prerequisites count: {len(prereqs)}")
            if prereqs:
                print(f"    - Prerequisite 1: {prereqs[0][:100]}...")

        else:
            print("\nNo CAPEC attack-patterns found or extracted.")

        # --- 4. SAVE THE EXTRACTED DATA ---
        # Check if any data was actually loaded before saving
        if any(capec_data.values()):
             save_for_rag(capec_data, OUTPUT_RAG_DIR)
        else:
             print("\nNo data loaded, skipping save step.")
        # --- End Save Step ---

    except Exception as e:
        print(f"\nAn error occurred during STIX processing: {e}")