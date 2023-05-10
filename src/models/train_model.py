import os

import DatasetClass
import EmotionNetTrain
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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

# создадим обратный вызов на раннюю остановку обучения
early_stop_callback = EarlyStopping(monitor="val_f1",
                                    patience=10,
                                    verbose=False,
                                    mode="max")

# создадим обратный вызов на сохранение чекпоинта модели,
# по интересующей нас метрике
# saves top-K checkpoints based on "val_loss" metric
ckpt_names = "check_point-{epoch:02d}-{val_loss:.2f}"
checkpoint_callback_val_loss = ModelCheckpoint(save_top_k=5,
                                               monitor="val_loss",
                                               mode="min",
                                               filename=ckpt_names)

# saves last-K checkpoints based on "global_step" metric
# make sure you log it inside your LightningModule
ckpt_names = "check_point-{epoch:02d}-{train_step}"
checkpoint_callback_last_step = ModelCheckpoint(
    save_top_k=1,
    monitor="train_step",
    mode="max",
    filename=ckpt_names,
)

# начинаем обучение модели
opt_lr = 1 * 10**-4
model = EmotionNetTrain.LitModel((3, 224, 224),
                                 emotion_num,
                                 fc_only=True,
                                 learning_rate=opt_lr)

# добавляем список обратных вызовов в параметр callbacks
trainer = pl.Trainer(accelerator="gpu",
                     callbacks=[early_stop_callback,
                                checkpoint_callback_val_loss,
                                checkpoint_callback_last_step],
                     max_epochs=2,
                     log_every_n_steps=10)

trainer.fit(model, dm)
