"""
Originally inspired by impl at https://github.com/microsoft/unilm/tree/master/beit

Modified by Haoyu Lu, for generating the spatial-temporal masked position for video diffusion transformer
"""

import random
import math
import numpy as np
import torch



class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, 
            min_aspect=0.3,):

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches

        max_aspect = 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

class VideoMaskGenerator:
    def __init__(self, input_size, spatial_mask_ratio=0.5):
        self.length, self.height, self.width = input_size

        self.spatial_generator = MaskingGenerator((self.height, self.width), spatial_mask_ratio * self.height * self.width)
        
        # idx = 0 Predict
        self.predict_given_frame_length = 8

        # idx = 1 Backward
        self.backward_given_frame_length = 8

        # idx = 2 Interpreation
        self.interpreation_step = 4

        # idx = 5 MLM ratio
        self.mlm_ratio = 0.8

    def __repr__(self):
        repr_str = "Generator(%d, %d, %d)" % (
            self.length, self.height, self.width)
        return repr_str

    def get_shape(self):
        return self.length, self.height, self.width

    def spatial_mask(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)


        start_idx = random.randint(0, 3)
        end_idx = random.randint(0, 3)

        spatial_mask = self.spatial_generator()
        # print("start_idx, end_idx", start_idx, end_idx)
        mask[start_idx:-end_idx] = spatial_mask
        return mask

    def temporal_mask(self, idx=0):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        # Predict
        if idx == 0:
            mask[self.predict_given_frame_length:] = 1
        # Backward
        elif idx == 1:
            mask[:-self.backward_given_frame_length] = 1
        # Interpreation
        elif idx == 2:
            mask = np.ones(shape=self.get_shape(), dtype=np.int)
            mask[::self.interpreation_step] = 0
        # Unconditional Generation
        elif idx == 3:
            mask = np.ones(shape=self.get_shape(), dtype=np.int)
        # Only one frames
        elif idx == 4:
            frame_idx = random.randint(0, mask.shape[0]-1)
            mask = np.ones(shape=self.get_shape(), dtype=np.int)
            mask[frame_idx] = 0
        # MLM
        else:
            for frame_idx in range(mask.shape[0]):
                if random.random() < self.mlm_ratio:
                    mask[frame_idx] = 1
        return mask

    def __call__(self, batch_size=1, device=None, idx=-1):
        if idx >= 0:
            if idx < 6:
                mask = self.temporal_mask(idx)
            else:
                mask = self.spatial_mask()
            return torch.tensor(mask).unsqueeze(0).repeat(batch_size,1,1,1).to(device)

        if random.random() < 0.2:
            mask = self.spatial_mask()
        else:
            idx = random.randint(0, 5)
            mask = self.temporal_mask(idx)
        
        return torch.tensor(mask).unsqueeze(0).repeat(batch_size,1,1,1).to(device)

if __name__ == '__main__':

    generator = VideoMaskGenerator((10,10,10))
    print(generator())
    mask = generator(4)
    print(mask.shape)

    a = torch.ones(32, 4, 10, 10, 10,)
    print((a[:] * mask).shape)