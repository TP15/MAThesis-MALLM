from openai import OpenAI
import json

# ✅ Setup OpenRouter client
client = OpenAI(
    api_key="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",  # Replace if needed
    base_url="https://openrouter.ai/api/v1"
)

# ✅ This version uses the prompt passed from the JSONL instead of hardcoded one
def generate_response(prompt: str, model="deepseek/deepseek-v3-base:free", temperature=0.7, max_tokens=300):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that should generate a plaintext description "
                        "of the input of the user. While describing, please mention that you have time."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # ✅ Print for debug and return clean output
        print("🧠 Model response:\n", response.choices[0].message.content)
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error: {e}"

# ✅ Process JSONL from your file
def process_jsonl(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile, start=1):
            try:
                data = json.loads(line)
                prompt = data.get("Output", "")
                if not prompt:
                    print(f"⚠️ Skipping line {idx}: No 'Output' field found.")
                    continue

                print(f"🔍 Processing line {idx}...")
                response = generate_response(prompt)

                result = {
                    "instruction": "",
                    "input": "",
                    "output": response.strip()
                }

                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON at line {idx}: {e}")
            except Exception as e:
                print(f"❌ Error processing line {idx}: {e}")

# ✅ Main script
if __name__ == "__main__":
    input_file = "/Users/thomaspathe/Documents/MAThesis-MALLM/categories.jsonl"
    output_file = "output_descriptions.jsonl"  
    process_jsonl(input_file, output_file)
