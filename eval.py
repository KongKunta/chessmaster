import re

import chess
from datasets import load_dataset

from agent import MODEL_NAME, generate_response

DATASET_NAME = "Lichess/chess-puzzles"

# 6 ratingintervaller x 5 puzzles = 30 puzzles
RATING_BUCKETS = [(0, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 2600), (2600, 4000)]
PER_BUCKET = 5

# Godkendte formater: "e2e4", "e7e8q" og "e2, e4" / "e2,e4" / "e2 e4"
# Alt andet tæller som forkert format
MOVE_RE = re.compile(r"([a-h][1-8]),?\s*([a-h][1-8])([qrbn]?)")


def parse_move(response: str):
    match = MOVE_RE.fullmatch(response.strip().lower())
    if match is None:
        return None
    try:
        return chess.Move.from_uci("".join(match.groups()))
    except ValueError:  # f.eks. "e2e2"
        return None


def evaluate(row):
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    board.push_uci(moves[0])  # modstanderens træk
    solution = moves[1]

    response = generate_response(board)
    move = parse_move(response)
    legal = move is not None and board.is_legal(move)
    correct = False
    if legal:
        board.push(move)
        # Lichess godkender også en anden mat-i-én end den i løsningen
        correct = move.uci() == solution or board.is_checkmate()
    return response, solution, move is not None, legal, correct


# Samme shuffle som train_chess.py. De første 20.000 bruges til træning, så dem springer vi over
dataset = load_dataset(DATASET_NAME, split="train")
candidates = dataset.shuffle(seed=0).select(range(20_000, 30_000))

puzzles = {bucket: [] for bucket in RATING_BUCKETS}
for row in candidates:
    for low, high in RATING_BUCKETS:
        if low <= row["Rating"] < high and len(puzzles[(low, high)]) < PER_BUCKET:
            puzzles[(low, high)].append(row)
    if all(len(rows) == PER_BUCKET for rows in puzzles.values()):
        break

print(f"Model: {MODEL_NAME}\n")
totals = [0, 0, 0]
n_total = 0
for (low, high), rows in puzzles.items():
    counts = [0, 0, 0]
    for row in rows:
        response, solution, fmt, legal, correct = evaluate(row)
        status = "KORREKT" if correct else "lovligt" if legal else "ulovligt" if fmt else "forkert format"
        print(f"{row['Rating']:>5}  svar: {response!r:<16} løsning: {solution}  {status}")
        for i, ok in enumerate((fmt, legal, correct)):
            counts[i] += ok
    n = len(rows)
    print(f"  {low}-{high}: format {counts[0]}/{n}, lovligt {counts[1]}/{n}, korrekt {counts[2]}/{n}\n")
    totals = [t + c for t, c in zip(totals, counts)]
    n_total += n

print(f"I alt ({n_total} puzzles)")
print(f"Rigtigt format: {totals[0]}/{n_total}")
print(f"Lovligt træk:   {totals[1]}/{n_total}")
print(f"Korrekt træk:   {totals[2]}/{n_total}")