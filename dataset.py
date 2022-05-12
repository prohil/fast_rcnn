"""
Python script to prepare the dataset
"""
import os
import numpy as np
import cv2
import torch
import glob
import albumentations as A
import pandas as pd
import config
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader


class PotHoleDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        # thermal_8_bit/FLIR_00010.jpeg
        records = self.df[self.df['image_id'] == image_id]
        image = cv2.imread(f"{self.image_dir}/thermal_8_bit/{image_id}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # convert the boxes into x_min, y_min, x_max, y_max format
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # get the area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        # we have only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # supposing that all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        # apply the image transforms
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            # convert the bounding boxes to PyTorch `FloatTensor`
            target['boxes'] = torch.stack(tuple(map(torch.FloatTensor,
                                                    zip(*sample['bboxes'])))).permute(1, 0)
        return image, target, image_id

    def __len__(self):
        return self.image_ids.shape[0]

def collate_fn(batch):
    """
    This function helps when we have different number of object instances
    in the batches in the dataset.
    """
    return tuple(zip(*batch))

# function for the image transforms
def train_transform():
    return A.Compose([
        # A.Flip(0.5),
        # A.RandomRotate90(0.5),
        # A.MotionBlur(p=0.2),
        # A.MedianBlur(blur_limit=3, p=0.1),
        # A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# path to the input root directory
DIR_INPUT = config.ROOT_PATH
df_path = config.TRAIN_ANNOTATIONS
# read the annotation CSV file
train_df = pd.read_csv(df_path)
print(train_df.head())
print(f"Total number of image IDs (objects) in dataframe: {len(train_df)}")
# get all the image paths as list
image_paths = glob.glob(f"{DIR_INPUT}/thermal_8_bit/*.jpeg")
image_names = []
for image_path in image_paths:
    image_names.append(image_path.split(os.path.sep)[-1].split('.')[0])
print(f"Total number of training images in folder: {len(image_names)}")
image_ids = train_df['image_id'].unique()
print(f"Total number of unique train images IDs in dataframe: {len(image_ids)}")
# number of images that we want to train out of all the unique images
train_ids = [image_name + ".jpeg" for image_name in image_names[:]]  # use all the images for training
train_df = train_df[train_df['image_id'].isin(train_ids)]
print(f"Number of image IDs (objects) training on: {len(train_df)}")

train_dataset = PotHoleDataset(train_df, DIR_INPUT, train_transform())
train_data_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)