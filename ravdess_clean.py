# Данный скрипт отбирает видео из всего датасета RAVDESS_frames
# по указанным параметрам

import os
from pathlib import Path
import shutil


extract_folder = 'RAVDESS_frames'
extension = '*.mp4'

# параметры отбора видео файлов
modality = {2}
vocal_channel = {1, 2}
emotion = set(range(1, 8 + 1))
emotional_intensity = {1, 2}
statement = {1, 2}
repetition = {1, 2}
actor = set(range(1, 24 + 1))
selection_by = [modality,
                vocal_channel,
                emotion,
                emotional_intensity,
                statement,
                repetition,
                actor]

# отбираем только нужные нам видео
for path in Path(extract_folder).rglob(extension):
    filename = path.name
    params = filename.split('.')[0].split('-')
    params = [int(param) for param in params]

    select_condition = all(
        [param in cond for param, cond in zip(params, selection_by)])

    if select_condition:
        new_path = os.path.join(extract_folder, filename)
        os.rename(path, new_path)
    else:
        os.remove(path)


# удаляем пустые папки
for file in os.scandir(extract_folder):
    if file.is_dir():
        shutil.rmtree(file.path)
