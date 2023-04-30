import glob
import os

import pandas as pd


def get_emotion_class(image_path):
    image_name = os.path.split(image_path)[-1].split('.')[0]
    emotion_class = int(image_name.split('-')[2])
    return emotion_class


def get_image_path(image_path):
    return os.sep.join(image_path.split(os.sep)[1:])


def create_csv_annot(datatset_path, image_ext, subset):
    search_path = os.path.join(datatset_path, subset, f'*{image_ext}')
    image_paths = glob.glob(search_path)
    image_paths.sort()
    image_paths = [get_image_path(image_path) for image_path in image_paths]
    image_classes = [get_emotion_class(image_path)
                     for image_path in image_paths]

    data = {'image_path': image_paths, 'emotion_class': image_classes}
    df = pd.DataFrame.from_dict(data)
    annot_name = f'{subset}.csv'
    annot_path = os.path.join(datatset_path, annot_name)
    df.to_csv(annot_path, index=False)


if __name__ == '__main__':
    datatset_path = 'RAVDESS_frames_set'
    image_ext = '.png'

    create_csv_annot(datatset_path, image_ext, 'train')
    create_csv_annot(datatset_path, image_ext, 'test')
