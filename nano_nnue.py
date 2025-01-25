import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import os, time
from tqdm import tqdm

from constants import *
from chessbench import ChessBench, collate as chessbench_collate

device = "cuda"

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
        indices_sample = F.gumbel_softmax(indices_logits, tau=1.0, hard=True)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=0.001, hard=True)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=1.0, hard=False)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=0.001, hard=False)

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

# TODO: investigate regularization
# TODO: validation with argmax probabilities

NUM_EPOCHS = 5
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
VALID_EVERY = 1000
VALID_BATCH_SIZE = 8
NUM_VALID_SAMPLES = 131072

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    train_dataset = ChessBench(os.path.join(curr_dir, "data/train/action_value@0004_data.bag"), sharded=True)
    valid_dataset = ChessBench(os.path.join(curr_dir, "data/test/action_value_data.bag"))
    small_valid_dataset = Subset(valid_dataset, range(NUM_VALID_SAMPLES))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=chessbench_collate)
    valid_dataloader = DataLoader(small_valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, collate_fn=chessbench_collate)

    checkpoint_dir = os.path.join(curr_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    torch.set_float32_matmul_precision('medium')

    model = NanoNNUE().to(device)
    optimizer = optim.Adam(model.parameters())

    @torch.compile()
    def train_step(boards_batch, probs_batch):
        model.train()
        grad_accum_steps = BATCH_SIZE // MICRO_BATCH_SIZE
        total_loss = 0.0
        optimizer.zero_grad()
        for i in range(grad_accum_steps):
            start = i * MICRO_BATCH_SIZE
            end = (i + 1) * MICRO_BATCH_SIZE
            micro_batch_boards = boards_batch[start:end].to(device)
            micro_batch_probs = probs_batch[start:end].to(device)
            loss = model.loss(micro_batch_boards, micro_batch_probs)
            lossf = loss.item()
            total_loss += lossf / grad_accum_steps
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return total_loss

    @torch.compile()
    def valid_loss():
        model.eval()
        total_val_loss = 0.0
        count = 0
        with torch.no_grad():
            for boards_batch, probs_batch in tqdm(valid_dataloader):
                boards_batch = boards_batch.to(device)
                probs_batch = probs_batch.to(device)
                loss = model.loss(boards_batch, probs_batch)
                total_val_loss += loss.item()
                count += 1
        avg_val_loss = total_val_loss / count if count else 0
        return avg_val_loss

    step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"starting epoch: {epoch}")
        for boards_batch, probs_batch in train_dataloader:

            st = time.monotonic()
            loss = train_step(boards_batch, probs_batch)
            elapsed = time.monotonic() - st
            print(f"step {step}: training loss: {loss} ({elapsed:.4f}s)")

            if step % VALID_EVERY == 0:
                vloss = valid_loss()
                print(f"==> validation loss at step {step}: {vloss}")
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{epoch}-{step}.pt")
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": vloss
                }, checkpoint_path)

            step += 1
