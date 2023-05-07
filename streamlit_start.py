import io
import streamlit as st
from PIL import Image

from emotion_net_class import EmotionNet

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    
model_path = 'my_model.pth'
model_dim = (3, 224, 224)
class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
emotion_net = EmotionNet(model_path, model_dim, class_labels)
    
st.title('Распознавание эмоций человека по фото')
image = load_image()
result = st.button('Распознать эмоцию')

if result:
    emotion_name, score = emotion_net.predict_on_image(image, return_label=True)
    st.write('Результаты распознавания эмоций:')
    st.write(f'{emotion_name} {score:.0%}')
