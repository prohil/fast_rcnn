import torch
import matplotlib
import matplotlib.pyplot as plt
import time
from model import model
import config
from engine import train, visualize, save_model
from dataset import train_data_loader, train_dataset

if config.DEBUG:
    visualize()

num_epochs = config.EPOCHS

for epoch in range(num_epochs):
    start = time.time()
    train_loss = train(train_data_loader)
    print(f"Epoch #{epoch} loss: {train_loss}")
    end = time.time()
    print(f"Took {(end - start) / 60} minutes for epoch {epoch}")

save_model()
