from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import PPOTrainer, PPOConfig

# Modell und Tokenizer laden
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Custom Reward Function
def custom_reward_function(output):
    # Beispiel: Belohne lange und kohärente Antworten
    length_reward = len(output.split()) / 10
    coherence_reward = 1.0 if "sinnvolle Antwort" in output else -1.0
    return length_reward + coherence_reward

# PPO Konfiguration
config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=32,
    ppo_epochs=10,
)

# PPO Trainer
trainer = PPOTrainer(model, tokenizer, config)

# Training
for batch in data_loader:
    input_ids = tokenizer(batch["text"], return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    reward = custom_reward_function(tokenizer.decode(outputs[0], skip_special_tokens=True))
    trainer.step(input_ids, outputs, reward)
