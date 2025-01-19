import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "cuda"

NUM_PIECES = 10 # 1 our pawn, 2 our knight, 3 our bishop, 4 our rook, 5 our queen, 6-10 other pieces
NUM_SQUARES = 64
NUM_VECTORS = NUM_SQUARES * NUM_SQUARES # 4096
NUM_FEATURES = NUM_PIECES * NUM_SQUARES * NUM_SQUARES # (other piece type, other piece location, our king location)
CHANNEL_ROT = 16

class Embedding(nn.Module):
    def __init__(self, vector_dim=256, random_seed=42):
        super(Embedding, self).__init__()
        torch.manual_seed(random_seed)

        self.vector_dim = vector_dim

        shared_vectors_gen = torch.randn(NUM_VECTORS, vector_dim, requires_grad=False)
        self.shared_vectors = []
        for piece_idx in range(NUM_PIECES):
            rot_offset = piece_idx * CHANNEL_ROT
            self.shared_vectors.append(torch.roll(shared_vectors_gen, shifts=rot_offset, dims=1)) # NUM_PIECES * (NUM_VECTORS x vector_dim) = NUM_FEATURES x vector_dim

        self.indices_logits = nn.Parameter(torch.rand(NUM_FEATURES, NUM_VECTORS))

    # TODO: inference should only use partial evaluation since usually only a couple features change
    def forward(self, active_features):
        '''
        Forward pass
        active_features: (batch_size, num_features)
        '''
        batch_size, _ = active_features.shape

        indices_logits = self.indices_logits.to(active_features.device)
        indices_logits = self.indices_logits.reshape(1, NUM_FEATURES, NUM_VECTORS).expand(batch_size, -1, -1)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=1.0, hard=True)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=0.001, hard=True)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=1.0, hard=False)
        indices_sample = F.gumbel_softmax(indices_logits, tau=0.001, hard=False)

        num_piece_features = NUM_FEATURES // NUM_PIECES
        position_embedding = torch.zeros(batch_size, self.vector_dim, device=active_features.device)
        for piece_idx in range(NUM_PIECES):
            start_idx = piece_idx * num_piece_features
            end_idx = (piece_idx + 1) * num_piece_features
            # TODO: i think the the three lines below can be combined into a single einsum if i change dims of piece_probs
            piece_probs = indices_sample[:, start_idx:end_idx, :].to(active_features.device)
            piece_shared_vectors = self.shared_vectors[piece_idx].to(active_features.device)
            piece_vectors = torch.einsum("bqv,vd->bd", piece_probs, piece_shared_vectors)
            position_embedding += piece_vectors

        return position_embedding

    def regularization_king_square(self):
        reshaped_logits = self.indices_logits.reshape(NUM_PIECES, NUM_SQUARES, NUM_SQUARES, NUM_VECTORS)
        return torch.var(reshaped_logits, dim=2).mean()

    def regularization_neighboring_squares(self):
        # This was completely taken from ChatGPT

        # Reshape logits to group square dimensions
        reshaped_logits = self.indices_logits.reshape(NUM_PIECES, NUM_SQUARES, NUM_SQUARES, NUM_VECTORS)

        # Define a 2D Laplacian kernel for computing differences
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 3, 3)

        # Apply convolution to capture differences
        conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        conv.weight.data = kernel
        conv.weight.requires_grad = False

        # Apply convolution to the logits
        logits_square_diff = conv(reshaped_logits[:, :, :, :1])  # Apply to the first channel only
        smoothness_loss = torch.mean(logits_square_diff ** 2)
        return smoothness_loss

    def regularization_index_penalty(self):
        return torch.mean(self.indices_logits ** 2)

    def regularization_loss(self, l1, l2, l3):
        return l1 * self.regularization_king_square() + l2 * self.regularization_neighboring_squares() + l3 * self.regularization_index_penalty()

class NanoNNUE(nn.Module):
    def __init__(self):
        super(NanoNNUE, self).__init__()
        self.half_kp = Embedding()
        self.seq = nn.Sequential(
            nn.Linear(512, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1) # TODO: maybe have 8, one for each game stage
        )

    def forward(self, features):
        us_embedding = self.half_kp.forward(features[:,0,:])
        them_embedding = self.half_kp.forward(features[:,1,:])
        total_embedding = torch.cat((us_embedding, them_embedding), dim=1)
        return self.seq.forward(total_embedding)

    def loss(self, x, y_exp, loss_fn=nn.MSELoss()):
        y = self.forward(x)
        regression_loss = loss_fn(y, y_exp)
        return regression_loss #+ self.half_kp.regularization_loss(0, 0, 1) # TODO: check other regularization as well

class Features:
    @staticmethod
    def feature_idx(piece, is_us, piece_pos, our_king_pos):
        # pos is (file, rank)
        # rank is [0,7], 0 is rank closest to player, 7 is rank furthest away from player
        # file is [0,7], 0 is file furthest left, 7 is rank furthest right
        piece_pos_idx = piece_pos[0] + piece_pos[1] * 8
        our_king_pos_idx = our_king_pos[0] * our_king_pos[1] * 8
        piece_map = dict(p=0, n=1, b=2, r=3, q=4) if is_us else dict(p=5, n=6, b=7, r=8, q=9)
        piece_num = piece_map[piece]
        return our_king_pos_idx + NUM_SQUARES * piece_pos_idx + (NUM_SQUARES ** 2) * piece_num

    @staticmethod
    def parse_fen(fen):
        board, turn, _, _, = fen.split(" ")
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

# TODO: tune over regularization parameters

BATCH_SIZE = 256
MICRO_BATCH_SIZE = 4

if __name__ == "__main__":
    from pyarrow import parquet as pq
    from tqdm import trange

    torch.set_float32_matmul_precision('high')

    model = NanoNNUE().to(device)
    optimizer = optim.Adam(model.parameters())

    for batch in pq.ParquetFile("train-00000-of-00124.parquet").iter_batches(batch_size=BATCH_SIZE):
        batch_boards, batch_evaluations = torch.tensor([], device="cpu"), torch.tensor([], device="cpu")
        for fen, cp, mate in zip(batch["fen"], batch["cp"], batch["mate"]):
            board = Features.encode_fen_to_features(fen.as_py()).to("cpu")
            if cp.as_py() is None: cp = mate
            cp = torch.tensor(cp.as_py() / 100.0).to("cpu")
            batch_boards = torch.cat((batch_boards, torch.unsqueeze(board, 0)))
            batch_evaluations = torch.cat((batch_evaluations, torch.unsqueeze(cp, 0)))

        model.train()

        while 1: # overfit the first batch
            @torch.compile
            def train_step():
                grad_accum_steps = BATCH_SIZE // MICRO_BATCH_SIZE
                total_loss = 0
                for i in (pbar:=trange(grad_accum_steps)):
                    micro_batch_boards = batch_boards[i*MICRO_BATCH_SIZE:(i+1)*MICRO_BATCH_SIZE,:,:].to(device)
                    micro_batch_evaluations = batch_evaluations[i*MICRO_BATCH_SIZE:(i+1)*MICRO_BATCH_SIZE].to(device)
                    loss = model.loss(micro_batch_boards, micro_batch_evaluations)
                    lossf = loss.item()
                    total_loss += lossf / grad_accum_steps
                    loss.backward()

                    pbar.set_postfix({"loss": f"{lossf:.4f}"})
                    pbar.update(1)
                    
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                print(f"Training loss: {total_loss}")
            train_step()
