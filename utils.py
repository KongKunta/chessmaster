import chess

PROMPT = """You are playing chess.

{side} to move.
{board}

Answer with exactly 4 characters: letter, number, letter, number.
The first square is where the piece is, the second is where it moves to.
Example: e2e4
Answer with nothing else."""


def expand_fen(fen):
    board, turn, castling, ep, *_ = fen.split()
    for d in "12345678":
        board = board.replace(d, "." * int(d))
    rows = [" ".join(r) for r in board.split("/")]
    return "\n".join(rows) + f"\n{turn} {castling} {ep}"


def build_prompt(board: chess.Board, tokenizer) -> str:
    side = "White" if board.turn == chess.WHITE else "Black"
    content = PROMPT.format(side=side, board=expand_fen(board.fen()))
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
