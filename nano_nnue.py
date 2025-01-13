import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NUM_PIECES = 11 # 1 our pawn, 2 our knight, 3 our bishop, 4 our rook, 5 our queen, 6-10 other pieces, 11 other king
NUM_SQUARES = 64
NUM_VECTORS = NUM_SQUARES * NUM_SQUARES # 4096
NUM_FEATURES = NUM_PIECES * NUM_SQUARES * NUM_SQUARES # (other piece, other piece, our king location)
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

        indices_logits = self.indices_logits.reshape(1, NUM_FEATURES, NUM_VECTORS).expand(batch_size, -1, -1)
        indices_sample = F.gumbel_softmax(indices_logits, tau=1.0, hard=True)

        num_piece_features = NUM_FEATURES // NUM_PIECES
        position_embedding = torch.zeros(batch_size, self.vector_dim, device=active_features.device)
        for piece_idx in range(NUM_PIECES):
            start_idx = piece_idx * num_piece_features
            end_idx = (piece_idx + 1) * num_piece_features
            piece_probs = indices_sample[:, start_idx:end_idx, :]
            piece_shared_vectors = self.shared_vectors[piece_idx]
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

    @staticmethod
    def encode_fen_to_features(position):
        '''
        position: fen string
        returns: tuple of black embedding and white embedding
        '''
        # TODO
        pass

    def forward(self, features):
        black_embedding = self.half_kp.forward(features[0])
        white_embedding = self.half_kp.forward(features[1])
        total_embedding = torch.cat((black_embedding, white_embedding), dim=1)
        return self.seq.forward(total_embedding)

    def loss(self, x, y_exp, loss_fn=nn.MSELoss):
        y = self.forward(x)
        regression_loss = loss_fn(y, y_exp)
        return regression_loss + self.half_kp.regularization_loss()

if __name__ == "__main__":
    from pyarrow import parquet as pq
    import pyarrow as pa

    model = NanoNNUE()
    optimizer = optim.Adam(model.parameters())

    for batch in pq.ParquetFile("train-00000-of-00124.parquet").iter_batches(batch_size=8):
        batch_board, batch_evaluations = torch.tensor([]), torch.tensor([])
        for fen, cp in zip(batch["fen"], batch["cp"]):
            board = NanoNNUE.encode_fen_to_features(fen)
            cp = torch.tensor(cp.as_py() / 100.0)
            batch_board = torch.cat((batch_board, torch.unsqueeze(board, 0)))
            batch_evaluations = torch.cat((batch_evaluations, torch.unsqueeze(cp, 0)))

        model.train()

        while 1: # NOTE: overfit the first batch
            loss = model.loss(batch_board, batch_evaluations)
            loss.backward()
            optimizer.step()
            print(f"Training loss: {loss}")
