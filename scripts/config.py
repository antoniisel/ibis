import torch
from pathlib import Path
import torch.optim as optim


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
method_name = "HTS"
target_TF = "LEF1"
division_factor = 1

cycle_nums = ["2",]

# Model configurations
num_blocks = [2,2,2,2]
input_size, output_size, hidden_size, sequence_length = 5, 1, 10, 40

# Training configurations
num_epochs = 5
batch_size = 32


