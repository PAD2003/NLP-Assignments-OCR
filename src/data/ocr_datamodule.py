from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from src.data.components.vietocr_aug import ImgAugTransform
from src.data.components.ocr_vocab import Vocab
from src.data.components.custom_aug.wrapper import Augmenter
from src.data.components.ocr_dataset import OCRDataset, OCRTransformedDataset, OCRCompleteDataset, ClusterRandomSampler, Collator
import datetime
import shutil

class OCRDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str, 
        gt_paths: list[str],
        upsampling: list[list],
        vocab = Vocab(),
        custom_augmenter = Augmenter(),
        p = [0.6, 0.2, 0.1, 0.1],
        basic_augmenter = ImgAugTransform(),
        image_height = 32, 
        image_min_width = 32, 
        image_max_width = 512,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        masked_language_model: bool = False
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.val_loader = None
    
    def prepare_data(self) -> None:
        now = datetime.datetime.now()
        self.images_epoch_folder_name = f"{now.date()}_{now.time()}"

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load datasets only if not loaded already
        train_gt_path, val_gt_path, test_gt_path = self.hparams.gt_paths
        if not self.data_train and not self.data_val and not self.data_test:
            self.train_dataset = OCRDataset(
                data_dir=self.hparams.data_dir,
                gt_path=train_gt_path
            )
            if self.hparams.upsampling:
                for us in self.hparams.upsampling:
                    txt_path, times = us[0], int(us[1])
                    for i in range(times):
                        us_train_dataset = OCRDataset(
                            data_dir=self.hparams.data_dir,
                            gt_path=txt_path
                        )
                        self.train_dataset = ConcatDataset(datasets=[self.train_dataset, us_train_dataset])

            self.val_dataset = OCRDataset(
                data_dir=self.hparams.data_dir,
                gt_path=val_gt_path
            )

            self.test_dataset = OCRDataset(
                data_dir=self.hparams.data_dir,
                gt_path=test_gt_path
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        self.train_transformed_dataset = OCRTransformedDataset(
            dataset = self.train_dataset,
            task = "train",
            images_epoch_folder_name = self.images_epoch_folder_name,
            vocab = self.hparams.vocab,
            custom_augmenter = self.hparams.custom_augmenter,
            p = self.hparams.p
        )
        self.train_completed_dataset = OCRCompleteDataset(
            dataset = self.train_transformed_dataset,
            basic_augmenter = self.hparams.basic_augmenter,
            image_height = self.hparams.image_height,
            image_min_width = self.hparams.image_min_width,
            image_max_width = self.hparams.image_max_width
        )
        return DataLoader(
            dataset = self.train_completed_dataset,
            batch_size = self.batch_size_per_device,
            shuffle = False,
            sampler = ClusterRandomSampler(self.train_completed_dataset, self.batch_size_per_device, True),
            num_workers = self.hparams.num_workers,
            collate_fn = Collator(self.hparams.masked_language_model),
            pin_memory = self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.val_loader is not None:
            return self.val_loader
        
        self.val_transformed_dataset = OCRTransformedDataset(
            dataset = self.val_dataset,
            task = "val",
            images_epoch_folder_name = self.images_epoch_folder_name,
            vocab = self.hparams.vocab,
            custom_augmenter = None
        )
        self.val_completed_dataset = OCRCompleteDataset(
            dataset = self.val_transformed_dataset,
            basic_augmenter = self.hparams.basic_augmenter,
            image_height = self.hparams.image_height,
            image_min_width = self.hparams.image_min_width,
            image_max_width = self.hparams.image_max_width
        )
        self.val_loader = DataLoader(
            dataset = self.val_completed_dataset,
            batch_size = self.batch_size_per_device,
            shuffle = False,
            sampler = ClusterRandomSampler(self.val_completed_dataset, self.batch_size_per_device, True),
            num_workers = self.hparams.num_workers,
            collate_fn = Collator(self.hparams.masked_language_model),
            pin_memory = self.hparams.pin_memory,
        )
        return self.val_loader

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        self.test_transformed_dataset = OCRTransformedDataset(
            dataset = self.test_dataset,
            task = "test",
            images_epoch_folder_name = self.images_epoch_folder_name,
            vocab = self.hparams.vocab,
            custom_augmenter = None
        )
        self.test_completed_dataset = OCRCompleteDataset(
            dataset = self.test_transformed_dataset,
            basic_augmenter = self.hparams.basic_augmenter,
            image_height = self.hparams.image_height,
            image_min_width = self.hparams.image_min_width,
            image_max_width = self.hparams.image_max_width
        )
        return DataLoader(
            dataset = self.test_completed_dataset,
            batch_size = self.batch_size_per_device,
            shuffle = False,
            sampler = ClusterRandomSampler(self.test_completed_dataset, self.batch_size_per_device, True),
            num_workers = self.hparams.num_workers,
            collate_fn = Collator(self.hparams.masked_language_model),
            pin_memory = self.hparams.pin_memory,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        folder_path = f"aug_epoch/{self.images_epoch_folder_name}"
        try:
            shutil.rmtree(folder_path)
            print(f"Deleted {folder_path}")
        except FileNotFoundError:
            print(f"{folder_path} do not exists")
        except Exception as e:
            print(f"Error: {str(e)}")

############################################################### TEST ###############################################################

import hydra
from omegaconf import DictConfig
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    ######### CREATE DATAMODULE #########
    # config
    data_dir = "data/OCR/new_train"
    gt_paths = [
        "data/OCR/gt_files/mini_10.txt", 
        "data/OCR/gt_files/mini_10.txt", 
        "data/OCR/gt_files/mini_10.txt"
    ]
    upsampling = [
        ["data/OCR/gt_files/mini_10.txt", 2]
    ]
    batch_size = 2

    # test datamodule
    dm = OCRDataModule(
        data_dir=data_dir,
        gt_paths=gt_paths,
        upsampling=upsampling,
        batch_size=2
    )
    dm.prepare_data()
    dm.setup()
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()
    num_datapoints = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    assert num_datapoints == 50
    batch = next(iter(dm.train_dataloader()))
    x, y = batch['img'], batch['tgt_output']
    print(f"Images tensor: {x.size()}")
    print(f"Output tensor: {y.size()}")
    dm.teardown()

    ######### INSTANTIATE DATAMODULE #########
    import hydra
    dm: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()
    num_datapoints = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    print(f"Length of train dataloader: {len(dm.train_dataloader())}")
    # assert num_datapoints == 50
    batch = next(iter(dm.train_dataloader()))
    x, y = batch['img'], batch['tgt_output']
    print(f"Images tensor: {x.size()}")
    print(f"Output tensor: {y.size()}")
    dm.teardown()


if __name__ == "__main__":
    main()

############################################################### TEST ###############################################################