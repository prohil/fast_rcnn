import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import model
from dataset import train_data_loader

# the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_pth = 'checkpoints/FLIR filtered/fasterrcnn_resnet50_fpn_filt_4ep_07_03.pth'
save_path = 'checkpoints/FLIR filtered/fasterrcnn_resnet50_fpn_filt_5ep_07_03.pth'
model = model().to(device)
if load_pth:
    model.load_state_dict(torch.load(load_pth))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)



def train(train_dataloader):
    model.train()
    running_loss = 0

    for i, data in enumerate(train_dataloader):

        optimizer.zero_grad()
        images, targets, images_ids = data[0], data[1], data[2]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        loss = sum(loss for loss in loss_dict.values())
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            print(f"Iteration #{i} loss: {loss}")

    train_loss = running_loss / len(train_dataloader.dataset)
    return train_loss

def save_model():
    torch.save(model.state_dict(), save_path)

def visualize():
    """
    This function will only execute if `DEBUG` is `True` in
    `config.py`.
    """
    images, targets, image_ids = next(iter(train_data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    for i in range(1):
        boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
        sample = images[i].permute(1,2,0).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        for box in boxes:
            cv2.rectangle(sample,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (220, 0, 0), 3)
        ax.set_axis_off()
        plt.imshow(sample)
        plt.show()