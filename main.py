import numpy as np
import pandas as pd
import torch
import os
from cv2 import imread, createCLAHE
from glob import glob
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import mean_squared_error, structural_similarity as ssim
from skimage.util import montage as montage2d
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.optim as optim

# Load data
all_xray_df = pd.read_csv('Data_Entry_2017_v2020.csv')
directory_pattern = 'images*/**/*.png'
all_image_paths = {os.path.basename(x): x for x in glob(directory_pattern, recursive=True)}
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

# Initial Data Visualization
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df.sample(3)

# Plot histograms
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
all_xray_df[['Follow-up #']].hist(ax=axs[0, 0])
axs[0, 0].set_title('Follow-up Number Distribution')
all_xray_df[['OriginalImagePixelSpacing[x']].hist(ax=axs[0, 1])
axs[0, 1].set_title('Original Image Pixel Spacing (x) Distribution')
all_xray_df[['Patient Age']].hist(ax=axs[1, 0])
axs[1, 0].set_title('Patient Age Distribution')
all_xray_df[['OriginalImage[Width']].hist(ax=axs[1, 1])
axs[1, 1].set_title('Original Image Width Distribution')
plt.tight_layout()
plt.show()

# Plot categorical data description
print(all_xray_df[['View Position', 'Patient Gender']].describe().T)

# Preprocess data
pa_xray_df = all_xray_df[all_xray_df['View Position'].isin(['PA']) & (all_xray_df['Patient Age'] < 120)].copy()
top_patients_df = pa_xray_df.groupby(['Patient ID']).count()[['Image Index']].reset_index().sort_values('Image Index', ascending=False).head(10)
pa_xray_df = pa_xray_df[pa_xray_df['Patient ID'].isin(top_patients_df['Patient ID'].values)].sort_values(['Patient ID', 'Follow-up #'])

# Plot Follow-up # vs Patient Age for the top patients
fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))
for p_id, c_rows in pa_xray_df.groupby('Patient ID'):
    ax1.plot(c_rows['Follow-up #'], c_rows['Patient Age'], '.-', label=p_id)
ax1.legend()
ax1.set_xlabel('Follow-up #')
ax1.set_ylabel('Patient Age')
ax1.set_title('Follow-up Number vs Patient Age for Top Patients')
plt.show()

# Define functions
OUT_DIM = (512, 512)
def simple_imread(im_path, apply_clahe=False):
    img_data = np.mean(imread(im_path), 2).astype(np.uint8)
    n_img = (255 * resize(img_data, OUT_DIM, mode='constant')).clip(0, 255).astype(np.uint8)
    if apply_clahe:
        clahe_tool = createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe_tool.apply(n_img)
    else:
        return n_img

# Define the model
class SimpleAffineRegistration(nn.Module):
    def __init__(self):
        super(SimpleAffineRegistration, self).__init__()
        self.theta = Parameter(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(2, 3))

    def stn(self, x, theta):
        theta_vec = theta.repeat((x.shape[0], 1, 1))
        grid = F.affine_grid(theta_vec, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def forward(self, x):
        return self.stn(x, self.theta)

# Visualization for multiple patients
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleAffineRegistration().to(device)

fig, axs = plt.subplots(len(top_patients_df), 6, figsize=(24, len(top_patients_df) * 4))

for i, patient_id in enumerate(top_patients_df['Patient ID'].values):
    patient_df = pa_xray_df[pa_xray_df['Patient ID'] == patient_id]
    scans = np.stack(patient_df['path'].map(simple_imread).values, 0)
    
    if len(scans) < 2:
        continue  # Skip if there are not enough scans

    moving_tensor = torch.tensor(np.expand_dims(np.expand_dims(scans[1], 0), -1) / 255.0, dtype=torch.float).to(device)
    fixed_tensor = torch.tensor(np.expand_dims(np.expand_dims(scans[0], 0), -1) / 255.0, dtype=torch.float).to(device)

    losses = []
    optimizer = optim.SGD(model.parameters(), lr=5e-3)
    for epoch in range(100):
        optimizer.zero_grad()
        torch_registered_scan = model(moving_tensor)
        loss = F.mse_loss(torch_registered_scan, fixed_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch % 10)==0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))
    print('Final Parameters', list(model.parameters()))

       

    registered_img = 255.0 * torch_registered_scan.detach().cpu().numpy()[0, :, :, 0]
    
    finding_1 = patient_df.iloc[0]['Finding Labels']
    finding_2 = patient_df.iloc[1]['Finding Labels']

    # # Generate montage for the patient's scans
    # montage_img = montage2d(scans)

    # # Display the montage in the first subplot
    # axs[i, 0].imshow(montage_img, cmap='bone')
    # axs[i, 0].set_title(f'Patient {patient_id} - Scans\n{finding_1}')
    # # fig, ax1 = plt.subplots(1,1, figsize = (12,12))
    # # ax1.imshow(montage2d(scans), cmap = 'bone')
    # # fig.savefig('overview.png', dpi = 300)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    xx, yy = np.meshgrid(range(scans[1].shape[0]), range(scans[1].shape[1]))
    test_pattern = (((xx % 40)>30)|((yy % 40)>30)).astype(np.float32)
    ax1.imshow(test_pattern, cmap = 'bone_r')
    ax1.set_title('Test Pattern')
    test_tensor = torch.tensor(np.expand_dims(np.expand_dims(test_pattern, 0), -1), dtype = torch.float)
    skew_pattern = model.forward(test_tensor).detach().numpy()[0,:,:,0]
    ax2.imshow(skew_pattern, cmap = 'bone_r')
    ax2.set_title('Registered Pattern')
    
    # Calculate metrics
    mse_before = mean_squared_error(scans[0], scans[1])
    ssim_before, _ = ssim(scans[0], scans[1], full=True, data_range=scans[1].max() - scans[1].min())
    mse_after = mean_squared_error(scans[0], registered_img)
    ssim_after, _ = ssim(scans[0], registered_img, full=True, data_range=registered_img.max() - registered_img.min())
    
    axs[i, 0].imshow(scans[0], cmap='bone', vmax=255)
    axs[i, 0].set_title(f'Patient {patient_id} - Scan #1\n{finding_1}')
    axs[i, 1].imshow(scans[1], cmap='bone', vmax=255)
    axs[i, 1].set_title(f'Patient {patient_id} - Scan #2\n{finding_2}')
    axs[i, 2].imshow(1.0 * scans[1] - scans[0], vmin=-100, vmax=100, cmap='RdBu')
    axs[i, 2].set_title(f'Difference\nMSE: {mse_before:.2f}, SSIM: {ssim_before:.2f}')
    axs[i, 3].imshow(registered_img, cmap='bone', vmax=255)
    axs[i, 3].set_title('Registered Scan #2')
    axs[i, 4].imshow(registered_img - scans[0], vmin=-100, vmax=100, cmap='RdBu')
    axs[i, 4].set_title(f'Post-Registration Difference\nMSE: {mse_after:.2f}, SSIM: {ssim_after:.2f}')
    axs[i, 5].text(0.5, 0.5, f'MSE Before: {mse_before:.2f}\nSSIM Before: {ssim_before:.2f}\nMSE After: {mse_after:.2f}\nSSIM After: {ssim_after:.2f}', 
                   horizontalalignment='center', verticalalignment='center', transform=axs[i, 5].transAxes)
    axs[i, 5].axis('off')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout(pad=3.0)
plt.show()
