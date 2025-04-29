import numpy as np # linear algebra
import tensorflow as tf # for tensorflow based registration
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # for pytorch-based registration
import os
from cv2 import imread, createCLAHE # read and equalize images
import SimpleITK as sitk
import cv2
from glob import glob
import matplotlib.pyplot as plt

all_xray_df = pd.read_csv('Data_Entry_2017_v2020.csv')
directory_pattern = 'images*/**/*.png'
all_image_paths = {os.path.basename(x): x for x in 
                   glob(directory_pattern, recursive=True)}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

all_xray_df[['Follow-up #', 'OriginalImagePixelSpacing[x', 'Patient Age', 'OriginalImage[Width']].hist(figsize = (10, 10))

all_xray_df[['View Position', 'Patient Gender']].describe().T

pa_xray_df = all_xray_df[all_xray_df['View Position'].isin(['PA'])].copy()
pa_xray_df = pa_xray_df[pa_xray_df['Patient Age']<120]
top_patients_df = pa_xray_df.groupby(['Patient ID']
                                    ).count()[['Image Index']].reset_index().sort_values('Image Index', ascending = False).head(100)
pa_xray_df = pa_xray_df[pa_xray_df['Patient ID'].isin(top_patients_df['Patient ID'].values)].sort_values(['Patient ID', 'Follow-up #'])
fig, ax1 = plt.subplots(1,1, figsize = (20, 20))
for p_id, c_rows in pa_xray_df.groupby('Patient ID'):
    ax1.plot(c_rows['Follow-up #'], c_rows['Patient Age'], '.-', label = p_id)
ax1.legend()
ax1.set_xlabel('Follow-up #')
ax1.set_ylabel('Patient Age')
pa_xray_df.head(10)

from skimage.transform import resize
OUT_DIM = (512, 512)
def simple_imread(im_path, apply_clahe = False):
    img_data = np.mean(imread(im_path), 2).astype(np.uint8)
    n_img = (255*resize(img_data, OUT_DIM, mode = 'constant')).clip(0,255).astype(np.uint8)
    if apply_clahe:
        clahe_tool = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe_tool.apply(n_img)
    else:
        return n_img
    
test_img = simple_imread(all_xray_df['path'].values[0])
plt.matshow(test_img, cmap = 'bone')

# grab the scans from the first patient
first_patient_df = pa_xray_df[pa_xray_df['Patient ID'].isin([pa_xray_df['Patient ID'].values[0]])]
print(first_patient_df.shape[0], 'scans found')
first_patient_df.head(10)

first_scans = np.stack(first_patient_df['path'].map(simple_imread).values,0)

# might as well show-em if we have em
from skimage.util import montage as montage2d
fig, ax1 = plt.subplots(1,1, figsize = (12,12))
ax1.imshow(montage2d(first_scans), cmap = 'bone')
fig.savefig('overview.png', dpi = 300)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleAffineRegistration(nn.Module):
    def __init__(self):
        super(SimpleAffineRegistration, self).__init__()
        self.theta = Parameter(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(2,3))
    # Spatial transformer network forward function
    def stn(self, x, theta):
        theta_vec = theta.repeat((x.shape[0], 1, 1))
        grid = F.affine_grid(theta_vec, x.size())
        align_corners=False
        return F.grid_sample(x, grid)

    def forward(self, x):
        # transform the input
        return self.stn(x, self.theta)

model = SimpleAffineRegistration().to(device)

moving_tensor = torch.tensor(np.expand_dims(np.expand_dims(first_scans[2],0), -1)/255.0, dtype = torch.float)
fixed_tensor = torch.tensor(np.expand_dims(np.expand_dims(first_scans[1],0), -1)/255.0, dtype = torch.float)

torch_registered_scan_2 = model.forward(moving_tensor)
t_img = 255.0*torch_registered_scan_2.detach().numpy()[0,:,:,0]

fig, ((ax1, ax2, ax3), (ax1a, ax4, ax5)) = plt.subplots(2, 3, figsize = (12, 8))
ax1.imshow(first_scans[1], cmap = 'bone', vmax = 255)
ax1.set_title('Scan #1')
ax2.imshow(first_scans[2], cmap = 'bone', vmax = 255)
ax2.set_title('Scan #2')
ax3.imshow(1.0*first_scans[2]-first_scans[1], vmin = -100, vmax = 100, cmap = 'RdBu')
ax3.set_title('Difference')
ax1a.imshow(first_scans[1], cmap = 'bone', vmax = 255)
ax1a.set_title('Scan #1')
ax4.imshow(t_img, cmap = 'bone', vmax = 255)
ax4.set_title('Registered Scan 2')
ax5.imshow(t_img-first_scans[1], vmin = -100, vmax = 100, cmap = 'RdBu')
ax5.set_title('Post Registration Difference')

optimizer = optim.SGD(model.parameters(), lr=5e-3)
for epoch in range(100):
    optimizer.zero_grad()
    torch_registered_scan_2 = model.forward(moving_tensor)
    loss = F.mse_loss(torch_registered_scan_2, fixed_tensor)
    loss.backward()
    optimizer.step()
    if (epoch % 10)==0:
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))
    print('Final Parameters', list(model.parameters()))

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
xx, yy = np.meshgrid(range(first_scans[1].shape[0]), range(first_scans[1].shape[1]))
test_pattern = (((xx % 40)>30)|((yy % 40)>30)).astype(np.float32)
ax1.imshow(test_pattern, cmap = 'bone_r')
ax1.set_title('Test Pattern')
test_tensor = torch.tensor(np.expand_dims(np.expand_dims(test_pattern, 0), -1), dtype = torch.float)
skew_pattern = model.forward(test_tensor).detach().numpy()[0,:,:,0]
ax2.imshow(skew_pattern, cmap = 'bone_r')
ax2.set_title('Registered Pattern')


fig, ((ax1, ax2, ax3), (ax1a, ax4, ax5)) = plt.subplots(2, 3, figsize = (12, 8))
ax1.imshow(first_scans[1], cmap = 'bone', vmax = 255)
ax1.set_title('Scan #1')
ax2.imshow(first_scans[2], cmap = 'bone', vmax = 255)
ax2.set_title('Scan #2')
ax3.imshow(1.0*first_scans[2]-first_scans[1], vmin = -100, vmax = 100, cmap = 'RdBu')
ax3.set_title('Difference')
ax1a.imshow(first_scans[1], cmap = 'bone', vmax = 255)
ax1a.set_title('Scan #1')
t_img = 255.0*torch_registered_scan_2.detach().numpy()[0,:,:,0]
ax4.imshow(t_img, cmap = 'bone', vmax = 255)
ax4.set_title('Registered Scan 2')
ax5.imshow(t_img-first_scans[1], vmin = -100, vmax = 100, cmap = 'RdBu')
ax5.set_title('Post Registration Difference')
