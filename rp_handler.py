from inference import Converter
import subprocess
import runpod
import logging
import os
import requests
from convex import ConvexClient

DEVICE = "cuda"  # or "cpu" if local testing without a GPU

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



CONVEX_URL = os.environ.get("CONVEX_URL")

client = ConvexClient(CONVEX_URL)



def download_video(url, save_path):
    """
    Downloads a video from the specified URL and saves it to the given path.

    Args:
        url (str): The URL of the video to download.
        save_path (str): The file path where the video will be saved.
    """
    response = requests.get(url, timeout=(5, 60), stream=True)
    response.raise_for_status()

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def upload_video_to_convex(url, file_path):
    """
    Uploads a video file to the specified Convex URL.

    Args:
        url (str): The Convex endpoint URL to upload the video to.
        file_path (str): The path of the video file to upload.
    
    """
    with open(file_path, 'rb') as file:
        headers = {'Content-Type': 'video/webm'}
        response = requests.post(url, headers=headers, data=file)
        response.raise_for_status()
    return response.json().get('storageId')


def update_remove_bg_status(project_id: str, status: str, file_id: str | None = None, enabled: bool = True):
    import json
    remove_bg_data = json.dumps({
        "fileId": file_id,
        "removeBgEnabled": enabled,
    })

    client.mutation("workerActions:workerUpdateRemoveBgData", {
        "projectId": project_id,
        "removeBgData": remove_bg_data,
    })

    client.mutation("workerActions:workerUpdateRemoveBgStatus", args={
        "projectId": project_id,
        "status": status
    })

def generate_upload_url() -> str:
    return client.mutation("workerActions:workerGenerateUploadUrl", {})

def mux_audio(input_video, processed_video, final_video):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", processed_video,
        "-c:v", "copy",
        "-c:a", "libvorbis",
        "-map", "1:v:0",
        "-map", "0:a:0",
        final_video
    ]
    subprocess.run(cmd, check=True)


def handler(event):
    logger.info("Worker start")

    video_url = event['input'].get("video_url")
    project_id = event['input'].get("project_id")

    if not video_url:
        raise ValueError("video_url is required")
    if not project_id:
        raise ValueError("project_id is required")

    try:
        # Update status to processing
        update_remove_bg_status(project_id, "processing")

        # Download input video
        logger.info(f"Downloading video from {video_url}")
        download_video(video_url, "input.mp4")

        # Process video with background removal
        logger.info("Starting background removal processing")
        converter = Converter("rvm_mobilenetv3.pt", DEVICE)
        converter.convert(
            input_source="input.mp4",
            output_type="video",
            output_composition="output_no_audio.webm",
            seq_chunk=8,
            progress=False
        )

        # Mux audio back into the processed video
        logger.info("Muxing audio into processed video")
        mux_audio("input.mp4", "output_no_audio.webm", "output.webm")

        # Generate upload URL and upload the processed video
        logger.info("Generating upload URL")
        upload_url = generate_upload_url()
        
        logger.info("Uploading processed video to Convex storage")
        storage_id = upload_video_to_convex(upload_url, "output.webm")

        # Update status to completed with the file ID
        update_remove_bg_status(project_id, "completed", file_id=storage_id)

        logger.info(f"Successfully processed video. Storage ID: {storage_id}")
        return {"storageId": storage_id, "status": "completed"}

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        # Update status to failed
        try:
            update_remove_bg_status(project_id, "failed")
        except Exception as update_error:
            logger.error(f"Failed to update status to failed: {update_error}")
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})