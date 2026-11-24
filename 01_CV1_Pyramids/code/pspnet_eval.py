import numpy as np
from pspnet import *
import torch.nn as nn
import matplotlib.pyplot as plt
import os

plt.rcParams['image.cmap'] = 'gist_gray'
from torch.utils.data import DataLoader
from cityscapes_dataset import Cityscapes


# Βρίσκω το απόλυτο path για το φάκελο cityscapes γιατί αλλιώς δε δούλευε
current_script_dir = os.path.dirname(os.path.abspath(__file__))
data_root_abs = os.path.abspath(os.path.join(current_script_dir, '..', 'cityscapes_dataset'))
data_list_abs = os.path.join(data_root_abs, 'list', 'cityscapes', 'fine_val.txt')
print(f"Looking for dataset at: {data_root_abs}")
print(f"Looking for list file at: {data_list_abs}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset_eval = Cityscapes(split='val',
                           data_root=data_root_abs,
                           data_list=data_list_abs)
eval_dataloader = DataLoader(dataset_eval, batch_size=1, shuffle=True, num_workers=0)
criterion = nn.CrossEntropyLoss(ignore_index=255)

model_path = os.path.join(current_script_dir, 'train_epoch_200_CPU.pth')
model = PSPNet(layers=50, bins=(2, 3, 6, 8), dropout=0.1, classes=35,
               zoom_factor=8, use_ppm=True, pretrained=False, criterion=criterion)

print(f"Loading weights from {model_path}...")
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

def calc_iou(pred, label, num_classes):
    pred = pred.flatten()
    label = label.flatten()

    mask = label != 255
    pred = pred[mask]
    label = label[mask]

    #Calculate confusion matrix
    intersect = pred[pred == label]
    area_intersect = np.histogram(intersect, bins=np.arange(num_classes + 1))[0]
    area_pred = np.histogram(pred, bins=np.arange(num_classes + 1))[0]
    area_label = np.histogram(label, bins=np.arange(num_classes + 1))[0]

    area_union = area_pred + area_label - area_intersect
    return area_intersect, area_union

num_classes = 35
total_intersect = np.zeros(num_classes)
total_union = np.zeros(num_classes)
iou_per_image_list = []

print("Starting evaluation loop...")

with torch.no_grad():
    for i, (img, mask) in enumerate(eval_dataloader):
        img = img.to(device)
        mask = mask.to(device)

        output = model(img)
        pred = torch.argmax(output, dim=1).cpu().numpy()
        mask = mask.cpu().numpy()

        intersect, union = calc_iou(pred, mask, num_classes)
        total_intersect += intersect
        total_union += union

        iou_per_image = intersect / (union + 1e-10)
        valid_classes = union > 0
        if valid_classes.sum() > 0:
            mean_iou_image = np.mean(iou_per_image[valid_classes])
            iou_per_image_list.append(mean_iou_image)
        if i % 10 == 0:
            print(f'Processed {i+1}/{len(eval_dataloader)} images')

#%% Final Statistics

iou_per_class = total_intersect / (total_union + 1e-10)
mean_iou = np.mean(iou_per_class)
var_iou = np.var(iou_per_class)
mean_iou_per_image = np.mean(iou_per_image_list)
var_iou_per_image = np.var(iou_per_image_list)

print(f"1. Στατιστικά Ανά Κλάση:")
print(f"Μέση Τιμή (mIoU): {mean_iou:.4f}")
print(f"Διασπορά (Variance): {var_iou:.4f}")

print(f"\n2. Στατιστικά Ανά Εικόνα:")
print(f"Μέση Τιμή: {mean_iou_per_image:.4f}")
print(f"Διασπορά: {var_iou_per_image:.4f}")