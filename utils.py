import chess

PROMPT = """You are playing chess. Reply with your move in UCI format (e.g. e2e4, or e7e8q for promotion) and nothing else.

{side} to move.
{board}"""


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
    
