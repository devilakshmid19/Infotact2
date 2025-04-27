import tarfile
import pandas as pd
import os

# Step 1: Set your paths
archive_path = r'C:\Users\devil\OneDrive\Desktop\infotact\project-3\AmazonMusicCompact.tar.xz'
extract_path = r'C:\Users\devil\OneDrive\Desktop\infotact\project-3\extracted'

# Step 2: Define safe_extract function with filter
def safe_extract(tar, path=".", filter=None, members=None, *, numeric_owner=False):
    # Use the filter argument to specify what gets extracted
    for member in tar.getmembers():
        if filter is None or filter(member):
            if member.isreg() or member.isdir():  # Only extract regular files and directories
                tar.extract(member, path, set_attrs=numeric_owner)

# Step 3: Extract the archive using safe_extract with a filter
if not os.path.exists(extract_path):
    os.makedirs(extract_path)

with tarfile.open(archive_path, 'r:xz') as tar:
    # Set a filter that allows you to control which files are extracted (optional)
    safe_extract(tar, path=extract_path, filter=None)

print("Archive extracted successfully.")

# Step 4: Load the CSV file
csv_path = os.path.join(extract_path, 'AmazonMusic', 'amazon_music_metadata.csv')

if os.path.exists(csv_path):
    train_data = pd.read_csv(csv_path)
    print("CSV loaded successfully.")
    print(train_data.head())  # Show first 5 rows
else:
    print("CSV file not found at:", csv_path)

