import json

def extract_assets(mal_file_path, output_file_path):
    with open(mal_file_path, 'r') as file:
        lines = file.readlines()

    categories = []
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
                categories.append(''.join(current_block).strip())
                inside_category = False

    with open(output_file_path, 'w') as outfile:
        for cat in categories:
            json.dump({'Output': cat}, outfile)
            outfile.write('\n')

    print(f"Extracted {len(categories)} categories to {output_file_path}")


extract_assets('MAL Languages/allMALfiles/sasLang.mal', 'categories.jsonl')