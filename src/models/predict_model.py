# Инференс модели
from EmotionNetInference import EmotionNet

model_path = 'my_model.pth'
model_dim = (3, 224, 224)
emotion_net = EmotionNet(model_path, model_dim)

path_to_image = 'RAVDESS_frames_set/test/02-02-03-02-01-02-24_frame_120_0.png'
emotion_cls, score = emotion_net.load_image_and_predict(path_to_image)

print(emotion_cls, score)
