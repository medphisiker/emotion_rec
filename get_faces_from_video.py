import glob
import os

import cv2
import mediapipe as mp
import tqdm


def get_bbox(img_height, img_width, detection):
    bbox = detection.location_data.relative_bounding_box
    height = int(bbox.height * img_height)
    width = int(bbox.width * img_width)
    xmin = int(bbox.xmin * img_width)
    ymin = int(bbox.ymin * img_height)

    return height, width, xmin, ymin


def runOnVideo(video, maxFrames):
    """Генератор кадров из видео. Продолжает генерировать кадры 
    пока не достигнуто количество кадров maxFrames.
    """

    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        yield frame

        readFrames += 1
        if readFrames > maxFrames:
            break


path_video_folder = 'RAVDESS_frames'
result_dataset_path = 'RAVDESS_frames_set'
every_frame = 1
subsets = {'train': set(range(1, 19 + 1)),
           'test': set(range(20, 24 + 1))}

# создаем каталог для хранения датасета из кадров видео
if not os.path.exists(result_dataset_path):
    os.makedirs(result_dataset_path)
    
# MediaPipe solution API
mp_face_detection = mp.solutions.face_detection

# Run MediaPipe Face Detection with short range model.
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5, model_selection=1)

video_paths = glob.glob(path_video_folder)

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
            img_height, img_width, channels = frame.shape

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            # сохраняем crop с лицами в датасет
            if results.detections:
                for idx, detection in enumerate(results.detections):
                    height, width, xmin, ymin = get_bbox(
                        img_height, img_width, detection)

                    frame_name = f'{video_name}_frame_{frame_num}_{idx}.png'
                    frame_path = os.path.join(result_dataset_path, frame_name)
                    cv2.imwrite(frame_path, frame[ymin:ymin+height, xmin:xmin+width])

        frame_num += 1

face_detection.close()
