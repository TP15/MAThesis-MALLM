# Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig

# Dein eigener Compiler als Reward-Funktion
import subprocess

def check_mal_compilation(mal_file_path):
    try:
        subprocess.run(["malc", mal_file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 1
    except subprocess.CalledProcessError:
        return -1
    
    # HOW TO USE   file_to_check = "/Users/thomaspathe/Documents/MAThesis-MALLM/MAL Languages/allMALfiles/exampleLang.mal"
    #result = check_mal_compilation(file_to_check)

# Hugging Face Model Setup
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# PPO (Proximal Policy Optimization) Konfiguration
config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=2
)
ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

# Trainingsdaten-Beispiele (Prompts)
prompts = [
    "Erstelle eine Funktion, die zwei Zahlen addiert:",
    "Implementiere eine Schleife, die von 0 bis 10 zählt:",
    # Weitere Prompts hier ergänzen...
]

# Reinforcement Learning Loop
for epoch in range(10):  # Anzahl der Trainings-Epochen
    print(f"\nEpoch {epoch+1}")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        response_ids = model.generate(**inputs, max_length=50)
        generated_code = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Reward berechnen (Compiler als Reward-Funktion)
        reward = check_mal_compilation(generated_code)
        print(f"Prompt: {prompt}\nGenerated Code: {generated_code}\nReward: {reward}")

        # PPO Update durchführen
        query_tensor = inputs['input_ids']
        response_tensor = response_ids[:, query_tensor.shape[-1]:]
        ppo_trainer.step(query_tensor, response_tensor, torch.tensor([reward]))

# Modell speichern nach Training
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
