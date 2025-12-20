from inference import Converter
import subprocess
from convex_helpers import download_video, upload_video_to_convex
import runpod
import logging
import os
from convex import ConvexClient

DEVICE = "cuda"  # or "cpu" if local testing without a GPU

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



CONVEX_URL = os.environ.get("CONVEX_URL")
WORKER_SECRET = os.environ.get("WORKER_SECRET")

client = ConvexClient(CONVEX_URL)

def update_remove_bg_status(project_id: str, status: str, file_id: str | None = None, enabled: bool = True):
    import json
    remove_bg_data = json.dumps({
        "fileId": file_id,
        "removeBgEnabled": enabled,
        "removeBgStatus": status,
    })
    client.mutation("workerActions:workerUpdateRemoveBgData", {
        "workerSecret": WORKER_SECRET,
        "projectId": project_id,
        "removeBgData": remove_bg_data,
    })

def generate_upload_url() -> str:
    return client.mutation("workerActions:workerGenerateUploadUrl", {
        "workerSecret": WORKER_SECRET,
    })

def mux_audio(input_video, processed_video, final_video):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", processed_video,
        "-c:v", "copy",
        "-c:a", "copy",
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
            output_composition="output_no_audio.mp4",
            seq_chunk=1,
            progress=False
        )

        # Mux audio back into the processed video
        logger.info("Muxing audio into processed video")
        mux_audio("input.mp4", "output_no_audio.mp4", "output.mp4")

        # Generate upload URL and upload the processed video
        logger.info("Generating upload URL")
        upload_url = generate_upload_url()
        
        logger.info("Uploading processed video to Convex storage")
        storage_id = upload_video_to_convex(upload_url, "output.mp4")

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