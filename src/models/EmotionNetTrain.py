import pytorch_lightning as pl
import torch
import torchvision.models as models
from torch import nn
from torchmetrics.classification import F1Score


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, classes_num, learning_rate=1e-4, fc_only=False):
        super().__init__()

        # метрика
        f1 = F1Score(task="multiclass", num_classes=classes_num)

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.step = 0

        # transfer learning if pretrained=True
        self.neural_net = models.resnet50(pretrained=True)
        self.neural_net.fc = nn.Linear(2048, classes_num)

        if fc_only:
            self.set_trainable_fc_only()

        self.criterion = nn.functional.cross_entropy
        self.metric = f1

    def set_trainable_fc_only(self):
        # freeze params всей нейросети
        for param in self.neural_net.parameters():
            param.requires_grad = False

        # размораживаем веса для fc
        for param in self.neural_net.parameters():
            param.requires_grad = True

    # will be used during inference
    def forward(self, x):
        return self.neural_net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        loss = self.criterion(y_pred, y)

        # training metrics
        preds = torch.argmax(y_pred, dim=1)
        metric_value = self.metric(preds, y)

        self.step += 1

        self.log("train_loss", loss)
        self.log("train_f1", metric_value)
        self.log("train_step", self.step)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        loss = self.criterion(y_pred, y)

        # training metrics
        preds = torch.argmax(y_pred, dim=1)
        metric_value = self.metric(preds, y)

        self.log("val_loss", loss)
        self.log("val_f1", metric_value)

        return loss

    def predict_step(self, batch, batch_idx):

        x, y = batch
        pred = self.forward(x)
        preds = torch.argmax(pred, dim=1)
        return preds.tolist()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        loss = self.criterion(y_pred, y)

        # training metrics
        preds = torch.argmax(y_pred, dim=1)

        return {"loss": loss, "outputs": preds, "gt": y}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'].float() for x in outputs]).mean()
        output = torch.cat([x['outputs'].float() for x in outputs], dim=0)

        gts = torch.cat([x['gt'].float() for x in outputs], dim=0)

        self.log("test_loss", loss)
        mse = self.metric(output, gts)
        self.log("test_f1", mse)

        self.test_gts = gts
        self.test_output = output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
