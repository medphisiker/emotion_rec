import torch
from PIL import Image
from torchvision import transforms


class EmotionNet:
    """Класс для инференса нейронной сети для предсказания эмоция человека.
    """

    def __init__(self,
                 model_path,
                 model_dim,
                 class_labels):
        """Инициализатор класса.

        Parameters
        ----------
        model_path : str
            путь к файлу с сериализованной моделью обученной нейросети
        model_dim : tuple
            shape картинки на которые обучена нейронная сеть
        class_labels : list
            list строк, каждая из которых содержит название для метки класса.
        """

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
        """Загружаем картинку

        Parameters
        ----------
        path2image : str
            путь к картинке и датасета

        Returns
        -------
        np.array
            2d np.array представляющий собой картинку считаную Pil
        """
        image = Image.open(path2image)
        image.load()
        return image

    def preprocess_image(self,
                         pil_image):
        """Выполняет предобработку изображения PIL для нейронной сети.

        Parameters
        ----------
        pil_image : np.array
            2d np.array представляющий собой картинку считаную PIL

        Returns
        -------
        torch.tensor
            batch из одной единственной картинки на которой 
            мы хоти сделать предсказание.
        """
        image = self.transform(pil_image)
        # создаем батч из одной картинки
        image = image[None, :, :, :]
        return image

    def predict_on_image(self,
                         image,
                         return_label=False):
        """Предсказываем на картинке считаной PIL.

        Parameters
        ----------
        image : np.array
            2d np.array представляющий собой картинку считаную PIL
        return_label : bool, optional
            флаг на возвращение названия класса в виде строки 
            вместо номера класса, by default False

        Returns
        -------
        (emotion_class, score)
            emotion_class : int (or str if return_label==True)
                номера класса или если return_label==True,
                названия класса в виде строки
            score : float
                вероятность принадлежности именно этому классу 
        """
        image = self.preprocess_image(image)

        with torch.no_grad():
            pred = self.neural_net(image)
            emotion_class = torch.argmax(pred, dim=1).tolist()[0]
            score = torch.softmax(pred, dim=1).flatten().tolist()[
                emotion_class]

            if return_label:
                emotion_class = self.class_labels[emotion_class]

        return emotion_class, score

    def load_image_and_predict(self,
                               path2image,
                               return_label=False):
        """Предсказываем на картинке расположенной по пути path2image.

        Parameters
        ----------
        path2image : str
            путь к картинке на которой мы хотим осуществить предсказание
        return_label : bool, optional
            флаг на возвращение названия класса в виде строки 
            вместо номера класса, by default False

        Returns
        -------
        (emotion_class, score)
            emotion_class : int (or str if return_label==True)
                номера класса или если return_label==True,
                названия класса в виде строки
            score : float
                вероятность принадлежности именно этому классу 
        """
        image = self.load_image(path2image)
        emotion_class, score = self.predict_on_image(image, return_label)

        return emotion_class, score
