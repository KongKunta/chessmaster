import chess
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import build_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "Lichess/chess-puzzles"

# Datasæt
dataset = load_dataset(DATASET_NAME, split="train")

# Tokenizer og model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def to_ids(row):
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    board.push_uci(moves[0])  # modstanderens træk

    prompt = tokenizer(build_prompt(board, tokenizer))["input_ids"]
    target = tokenizer(moves[1] + tokenizer.eos_token)["input_ids"]

    return {"input_ids": prompt + target,
            "labels": [-100] * len(prompt) + target}


dataset = dataset.shuffle(seed=0).select(range(20_000))  # start med et udsnit
dataset = dataset.map(to_ids, remove_columns=dataset.column_names)
print(dataset)