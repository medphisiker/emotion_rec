# Данный скрипт детектирует лица из `mediapipe` и нарезаает crop'ы лиц
# из него в набор лиц актеров изображающих различные эмоции.

import glob
import os

import cv2
import mediapipe as mp
import tqdm


def get_bbox(img_height,
             img_width,
             detection):
    """Переводит нормализованные координаты bbox от face detector'a
    из mediapipe в координаты bbox в пикселях изображения

    Parameters
    ----------
    img_height : int
        высота изображения в пикселях
    img_width : int
        ширина изображения в пикселях
    detection : mediapipe detections struct
        структура хранящая детекции лиц людей от face detector

    Returns
    -------
    (height, width, xmin, ymin)
        координаты bbox в пикселях изображения
    """
    bbox = detection.location_data.relative_bounding_box
    height = int(bbox.height * img_height)
    width = int(bbox.width * img_width)
    xmin = int(bbox.xmin * img_width)
    ymin = int(bbox.ymin * img_height)

    return height, width, xmin, ymin


def runOnVideo(video,
               max_frames):
    """Генератор кадров из видео. Продолжает генерировать кадры 
    пока не достигнуто количество кадров maxFrames.

    Parameters
    ----------
    video : cv2.VideoCapture object
        объект видео прочитанного OpenCV
    max_frames : int
        количество кадров после которого нужно остановить покадровое
        чтение видео.

    """
    read_frames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        yield frame

        read_frames += 1
        if read_frames > max_frames:
            break


path_video_folder = 'data/interim/RAVDESS_frames'
video_ext = '.mp4'
result_dataset_path = 'data/processed/RAVDESS_frames_set'
every_frame = 5
subsets = {'train': set(range(1, 19 + 1)),
           'test': set(range(20, 24 + 1))}

# создаем каталог для хранения датасета из кадров видео
if not os.path.exists(result_dataset_path):
    os.makedirs(result_dataset_path)

train_path = os.path.join(result_dataset_path, 'train')
if not os.path.exists(train_path):
    os.makedirs(train_path)

test_path = os.path.join(result_dataset_path, 'test')
if not os.path.exists(test_path):
    os.makedirs(test_path)

# MediaPipe solution API
mp_face_detection = mp.solutions.face_detection

# Run MediaPipe Face Detection with short range model.
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5,
                                                 model_selection=1)

search_path = os.path.join(path_video_folder,
                           f'*{video_ext}')

video_paths = glob.glob(search_path)

for path_video in video_paths:
    # Extract video properties
    video = cv2.VideoCapture(path_video)  # 'video-input.mp4'
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Enumerate the frames of the video
    video_frames_gen = runOnVideo(video, num_frames)
    frame_num = 1
    for frame in tqdm.tqdm(video_frames_gen, total=num_frames):
        if frame_num % every_frame == 0:
            video_name = os.path.split(path_video)[-1].split('.')[0]
            actor_num = int(video_name.split('-')[-1])
            img_height, img_width, channels = frame.shape

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            # сохраняем crop с лицами в датасет
            if results.detections:
                for idx, detection in enumerate(results.detections):
                    height, width, xmin, ymin = get_bbox(img_height,
                                                         img_width,
                                                         detection)

                    frame_name = f'{video_name}_frame_{frame_num}_{idx}.png'
                    frame_path = ''
                    if actor_num in subsets['train']:
                        frame_path = os.path.join(train_path, frame_name)
                    elif actor_num in subsets['test']:
                        frame_path = os.path.join(test_path, frame_name)

                    if frame_path:
                        cv2.imwrite(frame_path,
                                    frame[ymin:ymin+height, xmin:xmin+width])

        frame_num += 1

face_detection.close()
