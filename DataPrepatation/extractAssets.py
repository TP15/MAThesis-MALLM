import re
import json


def extract_assets(mal_file_path, output_file_path):
    with open(mal_file_path, 'r') as file:
        content = file.read()

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
            assets.append({'Output': asset_block})
            index = i
        else:
            index += 1

    with open(output_file_path, 'w') as outfile:
        for asset in assets:
            json.dump(asset, outfile)
            outfile.write('\n')

    print(f"Extracted {len(assets)} assets to {output_file_path}")

# Example usage
extract_assets('MAL Languages/allMALfiles/sasLang.mal', 'assets.jsonl')
