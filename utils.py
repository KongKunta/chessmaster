def expand_fen(fen):
    board, turn, castling, ep, *_ = fen.split()
    for d in "12345678":
        board = board.replace(d, "." * int(d))
    rows = [" ".join(r) for r in board.split("/")]
    return "\n".join(rows) + f"\n{turn} {castling} {ep}"