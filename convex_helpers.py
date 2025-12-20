import requests


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
        headers = {'Content-Type': 'video/mp4'}
        response = requests.post(url, headers=headers, data=file)
        response.raise_for_status()
    return response.json().get('storageId')




