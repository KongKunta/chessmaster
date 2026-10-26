import chess
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import build_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # skift til mappen med den trænede model senere

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

print(build_prompt(chess.Board(), tokenizer))
