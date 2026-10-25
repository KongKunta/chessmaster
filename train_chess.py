from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "Lichess/chess-puzzles"

# Datasæt
dataset = load_dataset(DATASET_NAME, split="train")
print(dataset)

# Tokenizer og model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print(model.config.model_type, f"{model.num_parameters() / 1e6:.0f}M parametre")