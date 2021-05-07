import os
from uuid import uuid4

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, points, transform=None):
        self.transform = transform
        self.points = points

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]
        

class LungsDataset(Dataset):
    def __init__(self, img_dir, msk_dir, transform=None, seed=4321, is_rgb=False):
        self.transform = transform
        self.ct_C = -600
        self.ct_W = 1200
        self.seed = seed
        self._is_rgb = is_rgb
        self.input_images, self.target_masks = self.get_images_masks(
            img_dir, msk_dir, is_rgb)

    def process_ct_image(self, img):
        low = self.ct_C - self.ct_W // 2
        high = self.ct_C + self.ct_W // 2
        img_clipped = np.clip(img, low, high)
        img_stand = (img_clipped - low) / self.ct_W
        return img_stand

    def grey2rgb(self, img_grey):
        return np.stack((img_grey,)*3, axis=-1)

    def get_images_masks(self, img_dir, mask_dir, is_rgb):
        images = []
        masks = []

        mask_niis = os.listdir(mask_dir)[:]
        for msk_name in mask_niis:
            img_name = msk_name[:msk_name.rfind('_')] + '.nii.gz'

            msk_nii_path = os.path.join(mask_dir, msk_name)
            img_nii_path = os.path.join(img_dir, img_name)

            msk_nib = nib.load(msk_nii_path)
            img_nib = nib.load(img_nii_path)

            msk_np = msk_nib.get_fdata()
            img_np = img_nib.get_fdata()

            for i in range(msk_np.shape[2]):
                if is_rgb:
                    img_ = self.grey2rgb(img_np[:, :, i])
                else:
                    img_ = img_np[:, :, i:i+1]
                images.append(self.process_ct_image(img_))
                masks.append(msk_np[:, :, i: i+1])

        np.random.seed(self.seed)
        ids = np.arange(len(images))
        np.random.shuffle(ids)

        return [images[x] for x in ids], [masks[x] for x in ids]

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        # uid = str(uuid4()).replace('-', '_')
        # return [f'{idx}_{uid}', image, mask]
        return [idx, image, mask]
