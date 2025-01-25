import torch
from torch.utils.data import Dataset
from bagz import BagFileReader, BagShardReader
from apache_beam import coders

from constants import *

decode = coders.TupleCoder((coders.StrUtf8Coder(), coders.FloatCoder())).decode

class Features:
    @staticmethod
    def feature_idx(piece, is_us, piece_pos, our_king_pos):
        # pos is (file, rank)
        # rank is [0,7], 0 is rank closest to player, 7 is rank furthest away from player
        # file is [0,7], 0 is file furthest left, 7 is rank furthest right
        piece_pos_idx = piece_pos[0] + piece_pos[1] * 8
        our_king_pos_idx = our_king_pos[0] + our_king_pos[1] * 8
        piece_map = dict(p=0, n=1, b=2, r=3, q=4) if is_us else dict(p=5, n=6, b=7, r=8, q=9)
        piece_num = piece_map[piece]
        return our_king_pos_idx + NUM_SQUARES * piece_pos_idx + (NUM_SQUARES ** 2) * piece_num

    @staticmethod
    def parse_fen(fen):
        split_fen = fen.split(" ")
        board, turn = split_fen[0], split_fen[1]
        rows = board.split("/")
        white_positions = dict()
        black_positions = dict()
        for rank, row in enumerate(reversed(rows)):
            file = 0
            for char in row:
                if char.isdigit():
                    file += int(char)
                elif char.isalpha():
                    is_white = char.isupper()
                    piece = char.lower()
                    pos = (file, rank)
                    file += 1
                    positions = white_positions if is_white else black_positions
                    positions.setdefault(piece, []).append(pos)
        return white_positions, black_positions, turn

    @staticmethod
    def encode_fen_to_features(fen):
        '''
        position: fen string
        returns: tuple of black embedding and white embedding
        '''
        white_positions, black_positions, turn = Features.parse_fen(fen)
        
        def flip_positions(color_positions):
            # flip positions for "them" perspective
            for piece, positions in color_positions.items():
                color_positions[piece] = [(7 - file, 7 - rank) for file, rank in positions]
            return color_positions

        if turn == "w":
            us = white_positions
            them = black_positions
        else:
            us = flip_positions(black_positions)
            them = flip_positions(white_positions)

        us_features, them_features = torch.zeros(NUM_FEATURES), torch.zeros(NUM_FEATURES)
        us_king_pos, them_king_pos = us["k"][0], them["k"][0]

        for piece, positions in us.items():
            if piece == "k": continue
            for pos in positions:
                # print(pos)
                # print(piece)
                # print(them_king_pos)
                us_idx = Features.feature_idx(piece, True, pos, us_king_pos)
                them_idx = Features.feature_idx(piece, False, pos, them_king_pos)
                us_features[us_idx] = 1.0
                them_features[them_idx] = 1.0

        for piece, positions in them.items():
            if piece == "k": continue
            for pos in positions:
                us_idx = Features.feature_idx(piece, False, pos, us_king_pos)
                them_idx = Features.feature_idx(piece, True, pos, them_king_pos)
                us_features[us_idx] = 1.0
                them_features[them_idx] = 1.0

        return torch.stack([us_features, them_features])

print(Features.parse_fen("8/8/2B3N1/5p2/6p1/6pk/4K2b/7r w - -"))
Features.encode_fen_to_features("8/8/2B3N1/5p2/6p1/6pk/4K2b/7r w - -")

class ChessBench(Dataset):
    def __init__(self, path, sharded=False):
        self.reader = BagShardReader(path) if sharded else BagFileReader(path)
        self.length = len(self.reader)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        record = self._reader[index]
        fen, prob = decode(record)
        board_tensor = Features.encode_fen_to_features(fen).to("cpu")
        prob_tensor = torch.tensor(prob, dtype=torch.float, device="cpu")
        return board_tensor, prob_tensor

def collate(batch):
    """
    batch is a list of (board_tensor, prob_tensor) samples.
    We stack each component along dim=0 to form a mini-batch.
    """
    boards = torch.stack([item[0] for item in batch], dim=0)  # [B, 8, 8] in this example
    probs = torch.stack([item[1] for item in batch], dim=0)   # [B]
    return boards, probs