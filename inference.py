import torch
import os
import logging
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

def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    logger.info(f"Starting video conversion for input: {input_source}")
    logger.debug(f"Parameters: input_resize={input_resize}, downsample_ratio={downsample_ratio}, "
                 f"output_type={output_type}, seq_chunk={seq_chunk}")
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'

    # Initialize transform
    logger.info("Initializing transform")
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
        logger.info(f"Using resize transform: {input_resize}")
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    logger.info("Initializing reader")
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
        logger.info(f"Using VideoReader for file: {input_source}")
    else:
        source = ImageSequenceReader(input_source, transform)
        logger.info(f"Using ImageSequenceReader for directory: {input_source}")
    reader = DataLoader(source, batch_size=seq_chunk, num_workers=0)
    
    # Initialize writers
    logger.info("Initializing writers")
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        logger.info(f"Output frame rate: {frame_rate}")
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
            logger.info(f"Composition writer initialized: {output_composition}")
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
            logger.info(f"Alpha writer initialized: {output_alpha}")
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
            logger.info(f"Foreground writer initialized: {output_foreground}")
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
            logger.info(f"Composition sequence writer initialized: {output_composition}")
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
            logger.info(f"Alpha sequence writer initialized: {output_alpha}")
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')
            logger.info(f"Foreground sequence writer initialized: {output_foreground}")

    # Inference
    logger.info("Starting inference")
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    logger.info(f"Using device: {device}, dtype: {dtype}")
    
    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            frame_count = 0
            for src in reader:

                dsr = downsample_ratio
                if dsr is None:
                    dsr = auto_downsample_ratio(*src.shape[2:])
                    logger.debug(f"Auto downsample ratio: {dsr}")


                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, dsr)

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_type == 'video':
                        #com = fgr * pha + bgr * (1 - pha)
                        com = fgr * pha
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
    
    logger.info("Video conversion completed successfully")


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
