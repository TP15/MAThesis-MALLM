import os
import jsonlines  # Make sure to install with: pip install jsonlines
import maltoolbox

maltoolbox.compile(
    '/Users/thomaspathe/Documents/MAThesis-MALLM/MAL Languages/allMALfiles',
    '/Users/thomaspathe/Documents/MAThesis-MALLM/MAL Languages/allMALfiles/compiled'
)

# Convert all .mal files to JSONL format using jsonlines
def convert_mal_to_jsonl(input_folder, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        for filename in os.listdir(input_folder):
            if filename.endswith('.mal'):
                mal_path = os.path.join(input_folder, filename)
                with open(mal_path, 'r', encoding='utf-8') as f:
                    mal_content = f.read()

                json_obj = {
                    "instruction": "",
                    "input": "",
                    "output": mal_content
                }
                print(json_obj)
                writer.write(json_obj)

    print(f"Done! JSONL written to {output_file}")


convert_mal_to_jsonl(
    '/Users/thomaspathe/Documents/MAThesis-MALLM/MAL Languages/allMALfiles',
    'output_new.jsonl'
)