import torch
import torch.nn as nn
import torch.nn.functional as F

# https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/551257
# https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#halfkp
# https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/

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
        indices_sample = F.gumbel_softmax(indices_logits, tau=1.0, hard=False)

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

    def regularization_loss_term(self, l1, l2, l3):
        return l1 * self.regularization_king_square() + l2 * self.regularization_neighboring_squares() + l3 * self.regularization_index_penalty()

class NanoNNUE(nn.Module):
    def __init__(self):
        super(NanoNNUE, self).__init__()
        self.black_halfkp = Embedding()
        self.white_halfkp = Embedding()
        self.seq = nn.Sequential(
            nn.Linear(512, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 8) 
        )

    @staticmethod
    def encode_fen_to_features(position):
        # TODO: return stage as well
        pass

    def forward(self, features, stage):
        # convert position to features
        black_embedding = self.black_halfkp.forward(features[0])
        white_embedding = self.white_halfkp.forward(features[1])
        total_embedding = torch.cat((black_embedding, white_embedding), dim=1)
        assert total_embedding.shape[1] == 512
        return self.seq.forward(total_embedding)

    def loss():
        pass

emb = NanoNNUE()
b = torch.randn(1, NUM_FEATURES)
w = torch.randn(1, NUM_FEATURES)
print(emb.forward((b, w), None).shape)

if __name__ == "__main__":
    from pyarrow import parquet as pq

    dataset = pq.ParquetFile("train-00000-of-00124.parquet").iter_batches(batch_size=8)
