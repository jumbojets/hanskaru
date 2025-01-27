NUM_PIECES = 10 # 1 our pawn, 2 our knight, 3 our bishop, 4 our rook, 5 our queen, 6-10 other pieces
NUM_SQUARES = 64
NUM_VECTORS = NUM_SQUARES * NUM_SQUARES # 4096
NUM_FEATURES = NUM_PIECES * NUM_SQUARES * NUM_SQUARES # (other piece type, other piece location, our king location)
CHANNEL_ROT = 1
