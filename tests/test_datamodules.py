from pathlib import Path

import pytest
import torch

# from src.data.mnist_datamodule import MNISTDataModule


# @pytest.mark.parametrize("batch_size", [32, 128])
# def test_mnist_datamodule(batch_size: int) -> None:
#     """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
#     attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
#     correctly match.

#     :param batch_size: Batch size of the data to be loaded by the dataloader.
#     """
#     data_dir = "data/"

#     dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
#     dm.prepare_data()

#     assert not dm.data_train and not dm.data_val and not dm.data_test
#     assert Path(data_dir, "MNIST").exists()
#     assert Path(data_dir, "MNIST", "raw").exists()

#     dm.setup()
#     assert dm.data_train and dm.data_val and dm.data_test
#     assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

#     num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
#     assert num_datapoints == 70_000

#     batch = next(iter(dm.train_dataloader()))
#     x, y = batch
#     assert len(x) == batch_size
#     assert len(y) == batch_size
#     assert x.dtype == torch.float32
#     assert y.dtype == torch.int64

from src.data.ocr_datamodule import OCRDataModule

@pytest.mark.parametrize("batch_size", [2, 16])
def test_ocr_datamodule(batch_size: int) -> None:
    data_dir = "data/OCR/new_train"
    gt_paths = [
        "data/OCR/gt_files/mini_10.txt", 
        "data/OCR/gt_files/mini_10.txt", 
        "data/OCR/gt_files/mini_10.txt"
    ]
    upsampling = [
        ["data/OCR/gt_files/mini_10.txt", 2]
    ]

    dm = OCRDataModule(
        data_dir=data_dir,
        gt_paths=gt_paths,
        upsampling=upsampling
    )
    dm.prepare_data()

    dm.setup()
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    assert num_datapoints == 50

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64