import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

import os
from PIL import Image
from src.models.ocr_module import OCRLitModule
import torch
from src.models.components.translate.predictor import Predictor
from tqdm import tqdm
import datetime

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_listImage(test_folder_path):
    image_files = sorted(os.listdir(test_folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    results = []

    for image_file in image_files:
        results.append(Image.open(os.path.join(test_folder_path, image_file)))

    return image_files, results

def write_txt_file(filenames_dict, tmp_path):
    with open(tmp_path, "w") as file:
        for key, value in filenames_dict.items():
            file.write(f"{key}\t{value}\n")

def write_file(filenames, labels, path):
    l = len(filenames)
    with open(path, "w") as file:
        for i in range(l):
            file.write(f"{filenames[i]} {labels[i]}\n")

def inference(cfg: DictConfig):
    assert cfg.ckpt_path

    log.info("Instantiating model")
    model = OCRLitModule.load_from_checkpoint(cfg.ckpt_path, map_location=torch.device(cfg.device))

    log.info("Instantiating predictor")
    predictor = Predictor(model=model.net, vocab=model.vocab)

    log.info("Inference")
    
    # read all batch and then inference
    # image_files, images = get_listImage(cfg.test_folder_path)
    # sents = predictor.predict_batch(imgs=images, device=cfg.device)

    # read each batch
    sents = []
    image_files = [file for file in os.listdir(cfg.test_folder_path) if file != ".gitkeep"]
    batchs = [image_files[i:i + cfg.batch_size] for i in range(0, len(image_files), cfg.batch_size)]
    for batch in tqdm(batchs):
        images = []
        for image_file in batch:
            images.append(Image.open(os.path.join(cfg.test_folder_path, image_file)))
        sents.extend(predictor.predict_batch(imgs=images, device=cfg.device))

    log.info("Writting output")
    now = datetime.datetime.now()
    write_file(filenames=image_files, labels=sents, path=(cfg.output_path + str(now) + ".txt"))

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    inference(cfg)

if __name__ == "__main__":
    main()