import glob
import os

import pandas as pd


def get_emotion_class(image_path):
    """Получить метку класса эмоции из названия картинки датасета.

    Parameters
    ----------
    image_path : str
        путь к картинке из датасета

    Returns
    -------
    int
        метку класса кодирующего эмоцию человека
    """
    image_name = os.path.split(image_path)[-1].split('.')[0]
    emotion_class = int(image_name.split('-')[2])
    return emotion_class


def get_image_path(image_path):
    """Получить путь без корневой папки

    Parameters
    ----------
    image_path : str
        путь к картинке из датасета

    Returns
    -------
    str
        путь к картинке без коневой папки
    """
    return os.sep.join(image_path.split(os.sep)[1:])


def create_csv_annot(datatset_path,
                     image_ext,
                     subset):
    """Создает файл аннотаций в формате csv-таблицы.

    Parameters
    ----------
    datatset_path : str
        путь к датасету из вырезанных кадров из видео
    image_ext : str
        расширение файлов картинок. Например, image_ext='.png'
    subset : 
        название подвыборки датасета. Например, 'train', 'val', 'test'
    """
    search_path = os.path.join(datatset_path,
                               subset,
                               f'*{image_ext}')

    image_paths = glob.glob(search_path)
    image_paths.sort()
    image_paths = [get_image_path(image_path) for image_path in image_paths]
    image_classes = [get_emotion_class(image_path)
                     for image_path in image_paths]

    data = {'image_path': image_paths,
            'emotion_class': image_classes}

    df = pd.DataFrame.from_dict(data)
    annot_name = f'{subset}.csv'
    annot_path = os.path.join(datatset_path, annot_name)
    df.to_csv(annot_path, index=False)


if __name__ == '__main__':
    datatset_path = 'data/processed/RAVDESS_frames_set'
    image_ext = '.png'

    create_csv_annot(datatset_path, image_ext, 'train')
    create_csv_annot(datatset_path, image_ext, 'test')
