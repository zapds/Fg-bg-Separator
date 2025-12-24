import logging
import os
from inference import Converter
import subprocess

DEVICE = "cuda"  # or "cpu" if local testing without a GPU


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def mux_audio(video_input, video_output):
    """
    Muxes audio from video_input into video_output.
    Assumes video_output exists. Creates a temp file and renames it.
    """
    if not os.path.isfile(video_input):
        return

    temp_output = video_output + ".temp.webm"
    
    # FFmpeg command: 
    # -i video_output (visuals)
    # -i video_input (audio source)
    # -map 0:v (take video from first input)
    # -map 1:a (take audio from second input)
    # -c:v copy (don't re-encode video)
    # -c:a libvorbis (encode audio to vorbis for webm compatibility)
    # -shortest (stop when the shortest stream ends)
    cmd = [
        'ffmpeg', '-y',
        '-i', video_output,
        '-i', video_input,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'libvorbis',
        '-shortest',
        temp_output
    ]
    
    try:
        logger.info(f"Muxing audio from {video_input} to {video_output}...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(temp_output, video_output)
        logger.info("Audio muxing successful.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Audio muxing failed (input might have no audio): {e}")
        if os.path.exists(temp_output):
            os.remove(temp_output)



if __name__ == '__main__':

    logger.info("Starting main execution")
    
    input_video = 'input.mp4'
    output_video = 'output.webm'

    converter = Converter("rvm_mobilenetv3.pt", DEVICE)
    converter.convert(
        input_source=input_video,
        output_type="video",
        output_composition=output_video,
        seq_chunk=2,
        progress=False
    )
    
    logger.info("Muxing audio")
    mux_audio(input_video, output_video)
    logger.info("Processing complete")
