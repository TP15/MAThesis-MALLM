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

def generate_response(prompt: str, model="mistralai/mistral-7b-instruct:free", temperature=0.7, max_tokens=1000):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in the Meta Attack Language (MAL), which is used to define domain-specific threat modeling languages for cybersecurity. Your task is to analyze MAL code and generate precise, exhaustive, and plain-text descriptions of its structure and semantics. These descriptions will be stored in a Retrieval-Augmented Generation (RAG) knowledge base, so they must follow best practices for technical documentation. Each description should: explain the purpose and function of the MAL code; describe all components, including assets, attack steps, defenses, and associations; clarify the logical flow and relationships between elements; highlight any domain-specific implications; use plain, professional language suitable for a technical audience; avoid unnecessary repetition or vague summaries. Be exhaustive but concise. Assume the reader has technical knowledge but may not be familiar with the specific MAL implementation.")
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

                result = {
                    "input": response,
                    "output": prompt,
                    "type": mal_type,
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
    output_folder = "/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep/MAL_RAG_Descriptions"
    os.makedirs(output_folder, exist_ok=True)

    input_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    print(f" Found {len(input_files)} JSONL files to process.")

    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_path = os.path.join(output_folder, f"{base_name}_processed.jsonl")
        print(f" Processing {base_name}...")
        process_jsonl(input_file, output_path)

    combined_output_path = os.path.join("/Users/thomaspathe/Documents/MAThesis-MALLM/LLM-Code/RAG/RAG-DataPrep", "MAL_RAG.jsonl")
    combine_jsonl_files(output_folder, combined_output_path)
    print(f"\n All files processed and combined into: {combined_output_path}")
