import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

import os, time, math
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

        # indices_sample = F.gumbel_softmax(indices_logits, tau=1.0, hard=True)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=0.001, hard=True)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=1.0, hard=False)
        # indices_sample = F.gumbel_softmax(indices_logits, tau=0.001, hard=False)
        indices_sample = F.gumbel_softmax(indices_logits, tau=0.1, hard=False)

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

    def index_entropy(self):
        p = F.softmax(self.indices_logits, dim=-1)
        entropy = - (p * (p + 1e-9).log()).sum(dim=-1).mean()
        return entropy

    def expected_index(self):
        p = F.softmax(self.indices_logits, dim=-1)
        idxs = torch.arange(NUM_VECTORS, device=p.device, dtype=p.dtype)
        expected_index = (p * idxs).sum(dim=-1)
        return expected_index.mean(), (expected_index ** 2).mean()

class NanoNNUE(nn.Module):
    def __init__(self, initial_step=0):
        super(NanoNNUE, self).__init__()
        self.half_kp = Embedding()
        self.seq = nn.Sequential(
            nn.Linear(512, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1) # TODO: maybe have 8, one for each game stage
        )
        self.schedule_train_params(initial_step)
    
    def init_weights(self, init_mode):
        with torch.no_grad():
            if init_mode == "lower_bias":
                slope_vector = -0.01 * torch.arange(NUM_VECTORS, device=self.half_kp.indices_logits.device, dtype=self.half_kp.indices_logits.dtype)
                self.half_kp.indices_logits.copy_(slope_vector.unsqueeze(0).expand(NUM_FEATURES, -1))
            elif init_mode == "uniform":
                nn.init.constant_(self.half_kp.indices_logits, 0.0)
            elif init_mode == "peaked":
                for f in range(NUM_FEATURES):
                    winner = torch.randint(0, NUM_VECTORS, (1,))
                    self.half_kp.indices_logits[f].fill_(-5.0)  # negative
                    self.half_kp.indices_logits[f, winner] = 5.0
            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")

    def forward(self, features):
        us_embedding = self.half_kp.forward(features[:,0,:])
        them_embedding = self.half_kp.forward(features[:,1,:])
        total_embedding = torch.cat((us_embedding, them_embedding), dim=1)
        return self.seq.forward(total_embedding)

    def loss(self, x, y_exp, loss_fn=nn.MSELoss()):
        y = self.forward(x)
        regression_loss = loss_fn(y, y_exp)
        entropy = self.half_kp.index_entropy()
        expected_index, index_penalty = self.half_kp.expected_index()
        total_loss = regression_loss + entropy * self.train_params["entropy_lambda"] + index_penalty * self.train_params["index_lambda"]
        penalties = {"regression": regression_loss, "entropy": entropy, "expected_index": expected_index}
        return total_loss, penalties

    def schedule_train_params(self, step):
        if step < 1000: # WARMUP
            # no lambda penalties
            self.train_params = {"entropy_lambda": 0, "index_lambda": 0}
        else:
            self.train_params = {"entropy_lambda": 1/50, "index_lambda": 1/16384}
        return self.train_params

NUM_EPOCHS = 5
BATCH_SIZE = 256
MICRO_BATCH_SIZE = 8
VALID_EVERY = 1000
VALID_BATCH_SIZE = 8

LOAD = None

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    train_dataset = ChessBench(os.path.join(curr_dir, "data/train/state_value_data.bag"))
    valid_dataset = ChessBench(os.path.join(curr_dir, "data/test/state_value_data.bag"))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=chessbench_collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True, collate_fn=chessbench_collate)

    checkpoint_dir = os.path.join(curr_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    torch.set_float32_matmul_precision('medium')

    model = NanoNNUE().to(device)

    if LOAD is not None:
        weights = torch.load(LOAD, weights_only=True)["model_state_dict"]
        model.load_state_dict(weights)
    else:
        model.init_weights(init_mode="lower_bias")

    optimizer = optim.Adam(model.parameters())

    total_steps = 25000
    warmup_steps = 1000
    base_lr = 1e-3
    min_lr = 1e-5
    def warmup_lambda(current_step: int): return float(current_step) / float(warmup_steps)
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

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
            loss, penalties = model.loss(micro_batch_boards, micro_batch_probs)
            lossf = loss.item()
            total_loss += lossf / grad_accum_steps
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return total_loss, penalties # works for now, because penalties only change with model update

    @torch.compile()
    def valid_loss():
        model.eval()
        total_val_loss = 0.0
        count = 0
        with torch.no_grad():
            for boards_batch, probs_batch in tqdm(valid_dataloader):
                boards_batch = boards_batch.to(device)
                probs_batch = probs_batch.to(device)
                loss, _ = model.loss(boards_batch, probs_batch)
                total_val_loss += loss.item()
                count += 1
        avg_val_loss = total_val_loss / count if count else 0
        return avg_val_loss

    step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"starting epoch: {epoch}")
        for boards_batch, probs_batch in train_dataloader:

            model.schedule_train_params(step)

            st = time.monotonic()
            loss, penalties = train_step(boards_batch, probs_batch)
            elapsed = time.monotonic() - st
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step {step}: regression loss: {penalties['regression']:.7f}, entropy penalty: {penalties['entropy']:.7f}, expected index: {penalties['expected_index']:.7f} (lr {current_lr:.7f}) ({elapsed:.4f}s)")

            if step % VALID_EVERY == 0 and step != 0:
                vloss = valid_loss()
                print(f"==> validation loss at step {step}: {vloss}")
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{epoch}-{step}.pt")
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": vloss
                }, checkpoint_path)

            step += 1
