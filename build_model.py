# build_model.py (run once)
import torch

from model import MattingNetwork


device = "cuda"  # script on CPU
model = MattingNetwork("mobilenetv3")
model.load_state_dict(torch.load("rvm_mobilenetv3.pth", map_location="cuda"))
model.eval()

scripted = torch.jit.script(model)
frozen = torch.jit.freeze(scripted)

frozen.save("rvm_mobilenetv3.pt")
