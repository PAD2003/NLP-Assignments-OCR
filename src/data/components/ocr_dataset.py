from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from src.data.components.vietocr_aug import ImgAugTransform
from src.data.components.ocr_vocab import Vocab
from torch.utils.data.sampler import Sampler
import random
import torch
import cv2
import numpy as np
from src.data.components.custom_aug.wrapper import Augmenter
import math
import shutil

def delete_contents_of_folder(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Deleted all contents of the folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting contents of the folder: {folder_path}")

class OCRDataset(Dataset):
    ''' This dataset only loads images from files into numpy arrays '''

    def __init__(
        self, 
        data_dir: str, 
        gt_path: str
    ):
        self.data_dir = data_dir
        self.samples = self.load_data(gt_path)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple:
        sample: dict = self.samples[index]
        filename = sample["filename"]
        word = sample["label"]

        # open & process image
        image_path = os.path.join(self.data_dir, filename)
        image = np.array(Image.open(image_path).convert("RGB"))

        return {'filename': filename, 'image': image, 'label': word}

    def load_data(self, gt_path) :
        samples = []

        with open(gt_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, word = parts
                sample_dict = {'filename': filename, 'label': word}
                samples.append(sample_dict)

        return samples

class OCRTransformedDataset(Dataset):
    ''' This dataset applies all custom transformations & augmentation to input images and encodes labels '''
    def __init__(
        self, 
        dataset: OCRDataset,
        task: str,
        images_epoch_folder_name: str,
        vocab = Vocab(),
        custom_augmenter = Augmenter(),
        p = [0.6, 0.2, 0.1, 0.1],
    ):
        self.dataset = dataset
        self.vocab = vocab
        self.task = task
        self.images_epoch_folder_name = images_epoch_folder_name

        self.custom_augmenter = custom_augmenter
        self.p = p
        
        delete_contents_of_folder(f"aug_epoch/{self.images_epoch_folder_name}/{self.task}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        filename = sample['filename']
        image = sample['image']
        word = sample['label']

        # process image
        try:
            os.mkdir(f"aug_epoch/{self.images_epoch_folder_name}")
            os.mkdir(f"aug_epoch/{self.images_epoch_folder_name}/{self.task}")
        except FileExistsError:
            pass
        
        folder_path = f"aug_epoch/{self.images_epoch_folder_name}/{self.task}"
        try:
            os.mkdir(folder_path)
        except FileExistsError:
            pass

        image_path = os.path.join(folder_path, f"{index}_{filename.strip().split('.')[0]}.png")
        if not os.path.exists(image_path):
            if self.custom_augmenter:
                image: np.array = self.custom_augmenter(image, word, 1, self.p)[0]
            cv2.imwrite(image_path, image)

        # encoding word
        label = self.vocab.encode(word)

        return {'filename': filename, 'image_path': image_path, 'label': label}

class OCRCompleteDataset(Dataset):
    ''' This dataset applies basic transform by VietOCR default '''
    def __init__(
        self, 
        dataset: OCRDataset,
        basic_augmenter = ImgAugTransform(),
        image_height = 32, 
        image_min_width = 32, 
        image_max_width = 512
    ):
        self.dataset = dataset

        self.basic_augmenter = basic_augmenter

        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

        self.build_cluster_indices()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        filename = sample['filename']
        image_path = sample["image_path"]
        image = cv2.imread(image_path)
        label = sample['label']

        if self.basic_augmenter:
            image: np.array = self.basic_augmenter(image)
        image: np.array = self.process_image(Image.fromarray(image), self.image_height, self.image_min_width, self.image_max_width)

        return {'filename': filename, 'image': image, 'label': label}
    
    def build_cluster_indices(self):
        self.cluster_indices = defaultdict(list)
        for i in tqdm(range(len(self.dataset)), "Building cluster indices ..."):
            bucket = self.get_bucket(i)
            self.cluster_indices[bucket].append(i)  
    
    def get_bucket(self, idx):
        sample = self.dataset[idx]
        image_path = sample["image_path"]
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        new_w, _ = self.resize(width, height, self.image_height, self.image_min_width, self.image_max_width)
        return new_w

    ####### TRANSLATE #######
    @staticmethod
    def resize(w, h, expected_height, image_min_width, image_max_width):
        new_w = int(expected_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w/round_to)*round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height

    @staticmethod
    def process_image(image, image_height=32, image_min_width=32, image_max_width=512):
        img = image.convert('RGB')
        w, h = img.size
        new_w, image_height = OCRCompleteDataset.resize(w, h, image_height, image_min_width, image_max_width)
        img = img.resize((new_w, image_height), Image.LANCZOS)
        img = np.asarray(img).transpose(2,0, 1)
        return img
    ####### TRANSLATE #######

class ClusterRandomSampler(Sampler):
    def __init__(
        self, 
        data_source, 
        batch_size, 
        shuffle = True
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_num_added_sample = 0

    def __iter__(self):
        batch_lists = []
        for _, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [self.fill_batch(batch) for batch in batches]

            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)

        return iter(lst)

    def __len__(self):
        return len(self.data_source) + self.total_num_added_sample

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def fill_batch(self, batch):
        if len(batch) == self.batch_size:
            return batch
        num_added_sample = self.batch_size - len(batch)
        self.total_num_added_sample += num_added_sample
        added_sample = random.choices(batch, k=num_added_sample)
        return batch + added_sample

class Collator(object):
    def __init__(self, masked_language_model=False):
        self.masked_language_model = masked_language_model
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, batch):
        filenames = []
        img = []
        target_weights = []
        tgt_input = [] ##target input
        max_label_len = max(len(sample['label']) for sample in batch)
        for sample in batch:
            filename = sample['filename']
            image = sample['image']
            label = sample['label']

            filenames.append(filename)
            img.append(self.transform(image).permute(1, 2, 0))

            label_len = len(label)
            
            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len,dtype=np.float32))))


        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights)==0

        rs = {
            'img': torch.stack(img, dim=0),
            'filenames': filenames,
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask)
        }   
        
        return rs