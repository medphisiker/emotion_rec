import glob
import os
import zipfile


# разархивируем весь датасет
ravdess_path = "/home/admin-gpu/Downloads/datasets/RAVDESS"
extension = '.zip'
extract_folder = '/home/admin-gpu/Downloads/datasets/RAVDESS_frames'

mask_path = os.path.join(ravdess_path, f'*{extension}')
zip_files = glob.glob(mask_path)

for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_folder)
