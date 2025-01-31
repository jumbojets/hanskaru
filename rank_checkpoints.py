import os, math
import torch

from nano_nnue import NanoNNUE

curr_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(curr_dir, "checkpoints")

model = NanoNNUE()

for file in os.listdir(checkpoints_dir):
    if file.endswith(".pt"):
        file = os.path.join(checkpoints_dir, file)
        checkpoint = torch.load(file, weights_only=False)
        loss = checkpoint["loss"]
        model.load_state_dict(checkpoint["model_state_dict"])
        entropy = model.half_kp.index_entropy()
        index_penalty = model.half_kp.expected_index_penalty()
        print(f"{file} -- {loss:.7f}, {entropy:.7f}, {math.sqrt(index_penalty):.7f}")
