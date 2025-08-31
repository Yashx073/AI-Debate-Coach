# Save this as download_demo.py in your project folder

import requests
import os

DEMO_VIDEO_URL = "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
FILE_NAME = "sample-mp4-file.mp4"

def download_demo_file():
    print("Downloading demo video...")
    try:
        response = requests.get(DEMO_VIDEO_URL, stream=True)
        response.raise_for_status() # Check for request errors

        with open(FILE_NAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Download complete! File saved as {FILE_NAME}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading file: {e}")
        print("Please check your internet connection or try a different URL.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_demo_file()