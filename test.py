import argparse
import os
import torch
from inference import convert_video
from model import MattingNetwork

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Path to input video')
parser.add_argument('--output_dir', default='output', help='Directory to save output videos')
args = parser.parse_args()

# Prepare paths
input_path = args.input
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Load model
model = MattingNetwork("mobilenetv3").eval().to("cpu")
model.load_state_dict(torch.load("rvm_mobilenetv3.pth", map_location="cpu"))

# Run inference and write outputs
convert_video(
    model,
    input_source=input_path,
    output_type="video",
    output_composition=os.path.join(output_dir, "composition.mp4"),
    output_alpha=os.path.join(output_dir, "alpha.mp4"),
    output_foreground=os.path.join(output_dir, "foreground.mp4"),
    output_video_mbps=4,
    seq_chunk=1,
    device="cpu",
    dtype=torch.float32
)
