from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from src.data.components.ocr_vocab import Vocab
from src.models.components.labelsmoothingloss import LabelSmoothingLoss
from src.models.vietocr_utils.download_ckpt import download_weights
# from src.models.components.cer import CER

class OCRLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        pretrain: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        vocab: Vocab
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.vocab = vocab
        self.net = net(vocab_size=len(vocab))
        if pretrain:
            print("USING PRETRAINED MODEL")
            weight_file = download_weights(pretrain)
            state_dict = torch.load(weight_file, map_location=torch.device(self.device))
            for name, param in self.net.named_parameters():
                if name not in state_dict:
                    print('{} not found'.format(name))
                elif state_dict[name].shape != param.shape:
                    print('{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                    del state_dict[name]
            try:
                self.net.load_state_dict(state_dict, strict=False)
            except:
                print("Do not use pretrained model")
            

        # loss function
        self.criterion = LabelSmoothingLoss(
            classes=len(vocab),
            padding_idx=vocab.pad,
            smoothing=0.1
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # # for averaging CER across batches
        # self.cer = CER()
        # self.train_cer = MeanMetric()
        # self.val_cer = MeanMetric()
        # self.test_cer = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        # self.val_cer_best = MinMetric()

    def forward(self, x: torch.Tensor, tgt_input, tgt_padding_mask) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, tgt_input, tgt_padding_mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_cer.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']
        output = self.forward(img, tgt_input= tgt_input, tgt_padding_mask=tgt_padding_mask)
        output = output.flatten(0,1)
        tgt_output = tgt_output.flatten(0,1)

        loss = self.criterion(output, tgt_output)
    
        return loss, output,tgt_output

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # # cer_score = self.cer(preds, targets)
        # # self.train_cer(cer_score)
        # # self.log("train/cer", self.train_cer, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # # cer_score = self.cer(preds, targets)
        # # self.val_cer(cer_score)
        # # self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

        # # cer_score = self.val_cer.compute()
        # # self.val_cer_best(cer_score)
        # # self.log("val/cer_best", self.val_cer_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    print("Here")