import torch

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
