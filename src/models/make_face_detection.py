import mediapipe as mp
import cv2


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


img_path = 'data/processed/RAVDESS_frames_set/02-01-01-01-01-01-01_frame_1.png'

# MediaPipe solution API
mp_face_detection = mp.solutions.face_detection

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1,
                                      circle_radius=1)

# Run MediaPipe Face Detection with short range model.
with mp_face_detection.FaceDetection(min_detection_confidence=0.5,
                                     model_selection=1) as face_detection:

    image = cv2.imread(img_path,
                       cv2.COLOR_BGR2RGB)

    img_height, img_width, channels = image.shape

    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if results.detections:
        annotated_image = image.copy()

        for detection in results.detections:
            height, width, xmin, ymin = get_bbox(img_height,
                                                 img_width,
                                                 detection)

            mp_drawing.draw_detection(annotated_image,
                                      detection)

        cv2.imwrite('tmp.png', annotated_image)

        cv2.imwrite('tmp_crop.png',
                    image[ymin:ymin+height, xmin:xmin+width])
