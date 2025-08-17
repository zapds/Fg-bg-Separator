import torch
from inference import convert_video
from model import MattingNetwork   # make sure this import works

# Load model architecture
model = MattingNetwork("mobilenetv3").eval().to("cpu")

# Load weights (state_dict)
model.load_state_dict(torch.load("rvm_mobilenetv3.pth", map_location="cpu"))

# Run inference
convert_video(
    model,
    input_source="input.mp4",
    output_type="video",
    output_composition="composition.mp4",
    output_alpha="alpha.mp4",
    output_foreground="foreground.mp4",
    output_video_mbps=4,
    seq_chunk=1,
    device="cpu",
    dtype=torch.float32
)
