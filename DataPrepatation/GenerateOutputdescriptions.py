from openai import OpenAI
import json

client = OpenAI(
    api_key="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",  # Replace if needed
    base_url="https://openrouter.ai/api/v1"
)

def generate_response(prompt: str, model="nvidia/llama-3.3-nemotron-super-49b-v1:free", temperature=0.7, max_tokens=300):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that should generate a plaintext description "
                        "of the input provided by the user. While describing, please mention that you have time."
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
        print(f"🧠 Response received: {output[:80]}{'...' if len(output) > 80 else ''}")
        return output

    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error: {e}"

def process_jsonl(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile, start=1):
            try:
                data = json.loads(line)
                prompt = data.get("Output", "")
                if not prompt.strip():
                    print(f" Skipping line {idx}: Empty 'Output' field.")
                    continue

                print(f"🔍 Processing line {idx}...")
                response = generate_response(prompt)

                result = {
                    "instruction": "",
                    "input": "",
                    "output": response
                }

                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"❌ JSON error on line {idx}: {e}")
            except Exception as e:
                print(f"❌ Processing error on line {idx}: {e}")

if __name__ == "__main__":
    input_file = "/Users/thomaspathe/Documents/MAThesis-MALLM/categories.jsonl"
    output_file = "output_descriptions.jsonl"
    process_jsonl(input_file, output_file)
