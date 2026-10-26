import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import build_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # skift til mappen med den trænede model senere

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

def generate_response(board: chess.Board, max_new_tokens=8) -> str:
    prompt = build_prompt(board, tokenizer)
    # chat-skabelonen indeholder allerede specialtokens, så de skal ikke tilføjes igen
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # et UCI-træk er kun få tokens
            do_sample=False,  # greedy: altid det mest sandsynlige træk
            pad_token_id=tokenizer.eos_token_id,
        )

    # kun de nye tokens efter prompten
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
    board = chess.Board()
    print("Svar:", generate_response(board))
