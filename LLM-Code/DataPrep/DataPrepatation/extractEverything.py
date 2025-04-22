import json
import re

import os

def process_all_mal_files(input_folder, output_folder):
    # Stelle sicher, dass der Output-Ordner existiert
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mal"):
            input_path = os.path.join(input_folder, filename)
            lang_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{lang_name}.jsonl")

            print(f"Processing {filename}...")
            extract_all_blocks(input_path, output_path)

    print("Alle .mal-Dateien wurden verarbeitet.")


def extract_assets(content):
    assets = []
    index = 0
    while index < len(content):
        if content.startswith("asset", index):
            start = index
            brace_open = content.find("{", index)
            if brace_open == -1:
                break

            brace_count = 1
            i = brace_open + 1
            while i < len(content) and brace_count > 0:
                if content[i] == "{":
                    brace_count += 1
                elif content[i] == "}":
                    brace_count -= 1
                i += 1

            asset_block = content[start:i].strip()
            assets.append({'Output': asset_block, 'Type': 'asset'})
            index = i
        else:
            index += 1
    return assets

def extract_all_blocks(mal_file_path, output_file_path):
    with open(mal_file_path, 'r') as file:
        content = file.read()
        lines = content.splitlines()

    outputs = []

    # --- Extract categories ---
    inside_category = False
    brace_count = 0
    current_block = []

    for line in lines:
        if 'category ' in line and not inside_category:
            inside_category = True
            brace_count = 0
            current_block = [line]
            brace_count += line.count('{') - line.count('}')
        elif inside_category:
            current_block.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                outputs.append({'Output': '\n'.join(current_block).strip(), 'Type': 'category'})
                inside_category = False

    # --- Extract assets using your new method ---
    outputs.extend(extract_assets(content))

    # --- Extract associations ---
    assoc_pattern = re.compile(r'(associations\s*\{[^{}]*\})', re.MULTILINE | re.DOTALL)
    associations = assoc_pattern.findall(content)
    outputs.extend([{'Output': a.strip(), 'Type': 'association'} for a in associations])

    # --- Write all to JSONL ---
    with open(output_file_path, 'w') as outfile:
        for item in outputs:
            json.dump(item, outfile)
            outfile.write('\n')

    print(f"Extracted {len(outputs)} total blocks to {output_file_path}")

# Example usage:
#extract_all_blocks(
 #   "/Users/thomaspathe/Documents/MAThesis-MALLM/MAL Languages/allMALfiles/sasLang.mal",
  #  "combined_output.jsonl"
#)

process_all_mal_files(
    "/Users/thomaspathe/Documents/MAThesis-MALLM/MAL Languages/allMALfiles",
    "jsonl_outputs"
)

