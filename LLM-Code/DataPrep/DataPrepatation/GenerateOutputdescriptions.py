from openai import OpenAI
import json
import random
import os
import glob
import shutil

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

asset_instructions = [
    "Convert the following cyber-attack description into Meta Attack Language code.",
    "Generate a Meta Attack Language snippet that models the scenario described below.",
    "Given this cyber incident, write its equivalent representation in Meta Attack Language.",
    "Translate this attack story into valid Meta Attack Language format.",
    "Write the Meta Attack Language structure for the following description.",
    "Represent the attack behavior below in your custom Meta Attack Language.",
    "Use Meta Attack Language to capture the steps described here.",
    "Model this cybersecurity attack using Meta Attack Language syntax.",
    "Write code in Meta Attack Language to describe the attack below.",
    "Convert the input into a structured Meta Attack Language format."
]

category_instructions = [
    "Define the following group of assets under a single category using Meta Attack Language.",
    "Create a category in Meta Attack Language that includes the assets listed below.",
    "Given these related assets, write a category definition in Meta Attack Language.",
    "Group the following assets into a category using the correct Meta Attack Language syntax.",
    "Write a Meta Attack Language snippet that declares a category containing these assets.",
    "Translate the grouping below into a category definition in Meta Attack Language.",
    "Use Meta Attack Language to define a category that encompasses the following assets.",
    "Model the following group of assets under a category using Meta Attack Language.",
    "Write the Meta Attack Language representation for a category including these assets.",
    "Create a valid Meta Attack Language definition for the described asset category."
]
association_instructions = [
    "Define the relationship between these two assets using an association in Meta Attack Language.",
    "Write a Meta Attack Language association that links the following assets.",
    "Model the connection described below as an association in Meta Attack Language.",
    "Use Meta Attack Language to describe the association between the listed assets.",
    "Create an association definition in Meta Attack Language for this asset interaction.",
    "Translate this asset linkage into a valid Meta Attack Language association.",
    "Write a Meta Attack Language snippet representing this association between entities.",
    "Represent the described interaction as a Meta Attack Language association.",
    "Define an association in Meta Attack Language using the connection details below.",
    "Generate a Meta Attack Language association linking the following asset types."
]

language_instructions = [
    "Design a complete Meta Attack Language model that includes asset definitions, grouped categories, and associations for the scenario below.",
    "Write a Meta Attack Language file from scratch that defines relevant assets, organizes them into categories, and models their interactions through associations.",
    "Translate the following cybersecurity system into a full Meta Attack Language specification, including asset classes, categories, and associations.",
    "Given the described infrastructure and threat scenario, generate a comprehensive MAL language definition, covering assets, categories, and associations.",
    "Write a Meta Attack Language snippet that models the entire threat landscape below, including asset types, logical groupings (categories), and how they are connected (associations).",
    "Create a complete .mal file using Meta Attack Language that defines assets, groups them into categories, and specifies their relationships.",
    "Convert the architectural and threat description below into a structured Meta Attack Language definition with all required elements: assets, categories, and associations.",
    "Generate a Meta Attack Language structure that models a threat scenario using proper categories for asset grouping and associations for inter-asset relations.",
    "Build a complete attack surface model using Meta Attack Language by defining custom assets, assigning them to categories, and describing their interactions via associations.",
    "Using Meta Attack Language, create a full language model that reflects the described environment. Ensure to include asset types, logical category groupings, and all necessary associations."
]

def generate_response(prompt: str, model="mistralai/mistral-7b-instruct:free", temperature=0.7, max_tokens=1000):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical security analyst and writer. Your task is to generate unstructured natural language descriptions that indirectly describe components of MAL (Modeling Attack Language) code, such as individual categories, assets, associations, attack steps, or attributes. These descriptions are not formal documentation — they are written in a more natural, real-world tone, similar to what might be found in cybersecurity incident reports, internal threat modeling documents, informal analyst notes, system architecture overviews, or security audit findings. Your goals: make the text realistic and informal, but factually aligned with the MAL code component. Include only information that is explicitly present in the code — never invent or interpret threat scenarios or system behavior beyond what is defined. The description should help a language model recognize and reconstruct the original MAL concept (e.g., asset, attack step, or relationship) from noisy, freeform input. Writing style rules: do not use bullet points or structured formatting. Mimic the tone and flow of real-world technical writing or internal security team communication. Use synonyms, varied sentence structure, and realistic phrasing to simulate real sources. Refer to identifiers as if they were mentioned in passing during a security assessment. You may mention relationships between components (e.g., 'X connects to Y' or 'A depends on B') as long as they directly reflect the MAL code. Component-specific guidance: for assets, describe what exists, its attributes, and what steps or associations are tied to it. For attack steps, describe possible actions or behaviors related to the asset. For associations, describe how two components are linked. For categories, describe what kinds of components it includes. Do not output or reference the MAL code directly. Do not mention 'MAL', 'language', 'modeling', or any meta-level concepts. The output should look like something a security professional might write naturally in a report or documentation."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        output = response.choices[0].message.content.strip()
        print(f" Response received: {output[:80]}{'...' if len(output) > 80 else ''}")
        return output

    except Exception as e:
        print(f" Error: {e}")
        return f"Error: {e}"

def process_jsonl(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile, start=1):
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    print(f"  Skipping line {idx}: Not a JSON object.")
                    continue

                prompt = data.get("Output")
                mal_type = data.get("Type")

                if prompt is None:
                    print(f"  Skipping line {idx}: 'Output' is None.")
                    continue
                if not isinstance(prompt, str) or not prompt.strip():
                    print(f"  Skipping line {idx}: 'Output' is empty or not a string.")
                    continue

                print(f"Processing line {idx}...")
                response = generate_response(prompt)

                if mal_type == "category":
                    instruction = random.choice(category_instructions)
                elif mal_type == "asset":
                    instruction = random.choice(asset_instructions)
                elif mal_type == "association":
                    instruction = random.choice(association_instructions)
                elif mal_type == "language":
                    instruction = random.choice(language_instructions)
                else:
                    instruction = "Convert the following input into Meta Attack Language format."

                result = {
                    "instruction": instruction,
                    "input": response,
                    "output": prompt
                }

                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"  JSON error on line {idx}: {e}")
            except Exception as e:
                print(f"  Processing error on line {idx}: {e}")

def combine_jsonl_files(folder_path: str, combined_file_path: str):
    with open(combined_file_path, "w", encoding="utf-8") as outfile:
        for filename in sorted(glob.glob(os.path.join(folder_path, "*.jsonl"))):
            with open(filename, "r", encoding="utf-8") as infile:
                shutil.copyfileobj(infile, outfile)

if __name__ == "__main__":
    input_folder = "/Users/thomaspathe/Documents/MAThesis-MALLM/HelperData/MAL Languages/allMALfiles/jsonl_outputs"
    output_folder = "/Users/thomaspathe/Documents/MAThesis-MALLM/HelperData/MAL Languages/allMALfiles/jsonl_output/outputgenerated_jsonl_files"
    os.makedirs(output_folder, exist_ok=True)

    input_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    print(f" Found {len(input_files)} JSONL files to process.")

    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_path = os.path.join(output_folder, f"{base_name}_processed.jsonl")
        print(f" Processing {base_name}...")
        process_jsonl(input_file, output_path)

    combined_output_path = os.path.join("/Users/thomaspathe/Documents/MAThesis-MALLM/HelperData/MAL Languages/allMALfiles/jsonl_outputs/", "combined_outputfinal.jsonl")
    combine_jsonl_files(output_folder, combined_output_path)
    print(f"\n All files processed and combined into: {combined_output_path}")
