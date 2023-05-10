import os

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def df_get_classes(df):
    """Получить число классов в нашем датасета

    Parameters
    ----------
    df : pandas.DataFrame
        pandas.DataFrame содержаний аннотации нашего датасета.

    Returns
    -------
    int
        число классов в нашем датасете
    """
    return len(df.emotion_class.unique())


def df_change_paths(df,
                    path,
                    path_col):
    """Изменить пути к картинкам относительно корневой папки
    датасета в df на пути относительно каталога проекта

    Parameters
    ----------
    df : pandas.DataFrame
        pandas.DataFrame содержаний аннотации нашего датасета.
    path : str
        путь к папке с датасетом относительно каталога проекта
    path_col : str
        название столбца, содержащего пути к картинкам в файле аннотаций.
    """
    df[path_col] = path + '/' + df[path_col]


def read_and_preprocess_annotation(path2datset,
                                   rel_path2_train):
    """Загружет и выполняет необходимую предобработку файлов аннотаций
    датасета.

    Parameters
    ----------
    path2datset : str
        путь к папке с датасетом относительно каталога проекта
    rel_path2_train : str
        относительный путь к файлу аннотации подвыборки нашего датасета.
        Например, 'train.csv'

    Returns
    -------
    pandas.DataFrame
        pandas.DataFrame содержаний аннотации нашего датасета,
        предобработанный для обучения модели-классификатора картинок.
    """
    path2train = os.path.join(path2datset, rel_path2_train)
    df_train = pd.read_csv(path2train)

    # меняем пути
    df_change_paths(df_train,
                    path2datset,
                    path_col='image_path')

    # чтобы метки классы начались с 0, а не с 1-го.
    df_train.emotion_class -= 1

    return df_train


class ClassificationDataset(Dataset):
    """
    Датасет с картинками, который производит изменения размера картинок,
    аугментации и преобразование в тензоры PyTorch
    """

    def __init__(self,
                 img_paths,
                 target,
                 mode,
                 rescale_size):
        """Инициализатор нашего класса.

        Parameters
        ----------
        img_paths : list
            массив строк содержащих названий к картинкам датасета
        target : str
            название столбца таблицы файла аннотаций, содержащих 
            значение целевой переменной.
        mode : str
            название подвыборки датасета, - 'train', 'val', 'test'.
        rescale_size : tuple
            shape изображения к которому будет произведен resize
            исходного изображения.

        Raises
        ------
        NameError
            Поднимает ошибку если название подвыборки не совпадает
            с одним из 3х возможных, - 'train', 'val', 'test'.
        """

        super().__init__()
        # список файлов для загрузки
        self.files = img_paths

        # изменяем размер картинок датасета на указанный
        self.rescale_size = rescale_size

        # режим работы
        self.mode = mode
        self.available_modes = ['train', 'val', 'test']

        if self.mode not in self.available_modes:
            print(
                f"{self.mode} is not correct; correct modes: {self.available_modes}")
            raise NameError

        self.len_ = len(self.files)

        if self.mode != 'test':
            self.target = target

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        """Загружаем картинку

        Parameters
        ----------
        file : str
            путь к картинке и датасета

        Returns
        -------
        np.array
            2d np.array представляющий собой картинку считаную PIL
        """
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose([
            transforms.Resize(self.rescale_size),
            transforms.ToTensor(),
            # mean и std для набора данных ImageNet на котором были обучены
            # предобученные сети из torchvision
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # transforms.Normalize делается для того, чтобы подогнать наши
        # данные под данные на которых были предобучены наши нейронные сети
        # из torchvision.models они были обученные  на наборе данных
        # ImageNet в документации по самому torchvision.models так же
        # сказано https://pytorch.org/docs/stable/torchvision/models.html

        # All pre-trained models expect input images normalized in the same way,
        # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
        # where H and W are expected to be at least 224. The images have to be
        # loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
        # and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:

        # И ниже код от самих разработчиков PyTorch
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])

        # трансформация с аугментацией для обучающей выборки средствами PyTorch
        transform_augment = transforms.Compose([
            transforms.Resize(size=self.rescale_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(hue=.1, saturation=.1),
            transforms.ToTensor(),
            # mean и std для набора данных ImageNet на котором были обучены
            # предобученные сети из torchvision
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        x = self.load_sample(self.files[index])
        # x = np.array(x / 255, dtype='float32') *см. примечание ниже

        # У нас тр режима датасета
        # DATA_MODES = ['train', 'val', 'test']
        # train - обучающая выборка на которой мы обучаем нейросеть
        # (есть картинки и ответы к ним)
        # val - валидационная выборка на которой мы тестируем как хорошо
        #  нейросеть обучилась! (есть картинки и ответы к ним)
        # test - тестовая выборка на которой мы предсказываем ответы для
        #  скора в соревновании (есть картинки ответов нет!)

        if self.mode == 'test':  # если тестовая выборка у нас нет ответов и
            x = transform(x)    # датасет не должен аугментировать картинки
            return x
        else:
            if self.mode == 'train':  # аугментируем обучающую выборку
                x = transform_augment(x)
            else:
                x = transform(x)  # не аугментируем валидационную выборку

            # для train или val выборок у нас есть ответы по классам
            y = self.target[index].item()

            return x, y


class ClassificationDataModule(pl.LightningDataModule):
    """Создает Lightning Datamodule, который будет загружать данные
    из нашего датасета.
    """

    def __init__(self,
                 df_train,
                 df_val,
                 df_test,
                 img_path_col,
                 target_col,
                 rescale_size,
                 batch_size,
                 num_workers):
        """_summary_

        Parameters
        ----------
        df_train : pandas.DataFrame
            pandas.DataFrame содержащий аннотацию для подвыборки train датасета
        df_val : pandas.DataFrame
            pandas.DataFrame содержащий аннотацию для подвыборки val датасета
        df_test : pandas.DataFrame
            pandas.DataFrame содержащий аннотацию для подвыборки test датасета
        img_path_col : str
            название столбца содержащего путь к картинке из датасета
        target_col : str
            название столбца содержащего значение целевой переменной
        rescale_size : tuple
            shape изображения к которому будет произведен resize
            исходного изображения.
        batch_size : int
            размер batch используемого при обучении нейросети
        num_workers : int
            число потоков осуществляющих загрузку данных из датасета
        """

        super().__init__()

        # задаем параметры даталоадера
        self.rescale_size = rescale_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # задаем данные датасета
        train_paths = df_train[img_path_col].values
        train_target = df_train[target_col].values
        val_paths = df_val[img_path_col].values
        val_target = df_val[target_col].values
        test_paths = df_test[img_path_col].values
        test_target = df_test[target_col].values

        self.train = ClassificationDataset(train_paths,
                                           train_target,
                                           mode='train',
                                           rescale_size=self.rescale_size)

        self.val = ClassificationDataset(val_paths,
                                         val_target,
                                         mode='val',
                                         rescale_size=self.rescale_size)

        self.test = ClassificationDataset(test_paths,
                                          test_target,
                                          mode='test',
                                          rescale_size=self.rescale_size)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
