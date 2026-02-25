import gdown
import os

FOLDER_ID = "1ZDa2qHMPxQSXxg-Bn-LnoXwi41ZoZI-r"

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("Downloading data from Google Drive...")
gdown.download_folder(
    id=FOLDER_ID,
    output="data/",
    quiet=False
)
print("Done! Data saved to data/")
