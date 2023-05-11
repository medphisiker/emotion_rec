import os

import DatasetClass
import EmotionNetTrain
import pytorch_lightning as pl


# сменим директорию на корневой каталог проекта
project_dir = os.path.split(os.path.dirname(__file__))[0]
project_dir = os.path.split(project_dir)[0]
os.chdir(project_dir)

# загружаем датасеты
path2datset = 'data/processed/RAVDESS_frames_set'
rel_path2_train = 'train.csv'
rel_path2_test = 'test.csv'

df_train = DatasetClass.read_and_preprocess_annotation(path2datset,
                                                       rel_path2_train)
df_test = DatasetClass.read_and_preprocess_annotation(path2datset,
                                                      rel_path2_test)


# опр. число различных эмоций в датасете
emotion_num = DatasetClass.df_get_classes(df_train)

# создаем загрузчик данных
dm = DatasetClass.ClassificationDataModule(df_train,
                                           df_test,
                                           df_test,
                                           img_path_col='image_path',
                                           target_col='emotion_class',
                                           rescale_size=(224, 224),
                                           batch_size=64,
                                           num_workers=12)

# опр. модель и трейнер
model = EmotionNetTrain.LitModel((3, 224, 224),
                                 emotion_num,
                                 fc_only=False)

trainer = pl.Trainer(accelerator="gpu",
                     log_every_n_steps=10)

# запустим процесс подбора learning rate
lr_finder = trainer.tuner.lr_find(model, dm)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.savefig('src/visualization/opt_learning_rate.png')

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
print(new_lr)
