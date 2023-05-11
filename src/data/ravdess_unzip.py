# разархивируем датасет RAVDESS с помощью данного скрипта

# Каталог датасета выглядит так:

# RAVDESS
# ├── Video_Song_Actor_01.zip
# ├── Video_Song_Actor_02.zip
# ├── Video_Song_Actor_03.zip
# ├── Video_Song_Actor_04.zip
# ├── Video_Song_Actor_05.zip
# ├── Video_Song_Actor_06.zip
# ├── Video_Song_Actor_07.zip
# ├── Video_Song_Actor_08.zip
# ├── Video_Song_Actor_09.zip
# ├── Video_Song_Actor_10.zip
# ├── Video_Song_Actor_11.zip
# ├── Video_Song_Actor_12.zip
# ├── Video_Song_Actor_13.zip
# ├── Video_Song_Actor_14.zip
# ├── Video_Song_Actor_15.zip
# ├── Video_Song_Actor_16.zip
# ├── Video_Song_Actor_17.zip
# ├── Video_Song_Actor_19.zip
# ├── Video_Song_Actor_20.zip
# ├── Video_Song_Actor_21.zip
# ├── Video_Song_Actor_22.zip
# ├── Video_Song_Actor_23.zip
# ├── Video_Song_Actor_24.zip
# ├── Video_Speech_Actor_01.zip
# ├── Video_Speech_Actor_02.zip
# ├── Video_Speech_Actor_03.zip
# ├── Video_Speech_Actor_04.zip
# ├── Video_Speech_Actor_05.zip
# ├── Video_Speech_Actor_06.zip
# ├── Video_Speech_Actor_07.zip
# ├── Video_Speech_Actor_08.zip
# ├── Video_Speech_Actor_09.zip
# ├── Video_Speech_Actor_10.zip
# ├── Video_Speech_Actor_11.zip
# ├── Video_Speech_Actor_12.zip
# ├── Video_Speech_Actor_13.zip
# ├── Video_Speech_Actor_14.zip
# ├── Video_Speech_Actor_15.zip
# ├── Video_Speech_Actor_16.zip
# ├── Video_Speech_Actor_17.zip
# ├── Video_Speech_Actor_18.zip
# ├── Video_Speech_Actor_19.zip
# ├── Video_Speech_Actor_20.zip
# ├── Video_Speech_Actor_21.zip
# ├── Video_Speech_Actor_22.zip
# ├── Video_Speech_Actor_23.zip
# └── Video_Speech_Actor_24.zip


import glob
import os
import zipfile


# разархивируем весь датасет
ravdess_path = "data/raw/RAVDESS"
extension = '.zip'
extract_folder = 'data/interim/RAVDESS_frames'

mask_path = os.path.join(ravdess_path, f'*{extension}')
zip_files = glob.glob(mask_path)

for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_folder)
