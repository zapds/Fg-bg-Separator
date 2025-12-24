import torch
import os
import logging
import subprocess
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    def __init__(self, model_path: str, device: str):
        self.device = device
        logger.info(f"Loading model from {model_path} on device {device}")
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def convert(self, *args, **kwargs):
        logger.info("Starting conversion")
        convert_video(
            self.model,
            device=self.device,
            dtype=torch.float32,
            *args,
            **kwargs
        )

def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'png_sequence',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    
    logger.info(f"Starting video conversion for input: {input_source}")
    
    # ... (Asserts remain the same) ...
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'

    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    
    reader = DataLoader(source, batch_size=seq_chunk, num_workers=0)
    
    frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
    
    # Initialize writers
    if output_type == 'video':
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        
        if output_composition is not None:
            # Ensure output filename ends in .webm if we want transparency
            if not output_composition.lower().endswith('.webm'):
                logger.warning("Output composition filename does not end in .webm. Transparency might be lost.")
            
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))

        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
                
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        # ... (ImageSequenceWriter logic remains same) ...
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            frame_count = 0
            for src in reader:

                dsr = downsample_ratio
                if dsr is None:
                    # Assuming auto_downsample_ratio is imported or defined elsewhere in actual code
                    # If not, add import: from inference_utils import auto_downsample_ratio
                    try:
                        from inference_utils import auto_downsample_ratio
                        dsr = auto_downsample_ratio(*src.shape[2:])
                    except ImportError:
                        dsr = 1.0 # Fallback

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, dsr)

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                    
                if output_composition is not None:
                    if output_type == 'video':
                        # UPDATED LOGIC FOR WEBM TRANSPARENCY
                        # Stack Foreground and Alpha to create RGBA: [T, 4, H, W]
                        com = torch.cat([fgr, pha], dim=2) 
                    else:
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                        
                    writer_com.write(com[0])
                    del com
                
                frame_count += src.size(1)
                bar.update(src.size(1))
                del src, fgr, pha
            
            logger.info(f"Processed {frame_count} frames")
    

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise
    finally:
        # Clean up
        logger.info("Cleaning up writers")
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()