import gdown
import os
import zipfile

FOLDER_ID = "1ZDa2qHMPxQSXxg-Bn-LnoXwi41ZoZI-r"

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("Downloading data from Google Drive...")
gdown.download_folder(
    id=FOLDER_ID,
    output="data/raw/",
    quiet=False
)

# Unzip any zip files in data/raw/
for root, dirs, files in os.walk("data/raw/"):
    for file in files:
        if file.endswith(".zip"):
            zip_path = os.path.join(root, file)
            print(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall("data/raw/")
            os.remove(zip_path)
            print("Done!")

print("All data ready in data/raw/")
