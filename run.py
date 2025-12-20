import logging
from inference import Converter
import subprocess

DEVICE = "cuda"  # or "cpu" if local testing without a GPU


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



if __name__ == '__main__':

    logger.info("Starting main execution")
    
    input_video = 'input.mp4'
    processed_video = 'output_no_audio.mp4'
    output_video = 'output.mp4'

    converter = Converter("rvm_mobilenetv3.pt", DEVICE)
    converter.convert(
        input_source="input.mp4",
        output_type="video",
        output_composition="output_no_audio.mp4",
        seq_chunk=1,
        progress=False
    )

    logger.info("Adding audio to output video")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", processed_video,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "1:v:0",
        "-map", "0:a:0",
        output_video
    ]
    subprocess.run(cmd, check=True)
    logger.info("Processing complete")
