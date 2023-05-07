import torch
from PIL import Image
# your favorite machine learning tracking tool
from torchvision import transforms

class EmotionNet:
    def __init__(self, model_path, model_dim, class_labels):
        self.neural_net = torch.load(model_path)
        self.neural_net.eval()
        self.dim = model_dim
        self.class_labels = class_labels
        
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        self.transform = transforms.Compose([
            transforms.Resize(model_dim[1:]),
            transforms.ToTensor(),
            # mean и std для набора данных ImageNet на котором были обучены 
            # предобученные сети из torchvision
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

    @staticmethod
    def load_image(path2image):
        image = Image.open(path2image)
        image.load()
        return image

    def preprocess_image(self, pil_image):
        image = self.transform(pil_image)
        # создаем батч из одной картинки
        image = image[None, :, :, :]
        return image
    
    def predict_on_image(self, image, return_label=False):
        image = self.preprocess_image(image)
              
        with torch.no_grad():
            pred = self.neural_net(image)
            emotion_class = torch.argmax(pred, dim=1).tolist()[0]
            score = torch.softmax(pred, dim=1).flatten().tolist()[emotion_class]
            
            if return_label:
                emotion_class = self.class_labels[emotion_class]
        
        return emotion_class, score
    
    def load_image_and_predict(self, path2image, return_label=False):
        image = self.load_image(path2image)
        emotion_class, score = self.predict_on_image(image, return_label)
        
        return emotion_class, score
        
