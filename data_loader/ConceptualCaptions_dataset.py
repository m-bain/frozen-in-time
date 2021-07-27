import os
import zlib

import pandas as pd

from base.base_dataset import TextImageDataset


class ConceptualCaptions3M(TextImageDataset):
    """
    Conceptual Captions dataset. Split files are specific to my download regime.
    """

    def _load_metadata(self):
        # download specific
        split_files = {
            'train': 'Train_GCC-training-success-020321-VALIDIMG.csv',
            'val': 'Validation_GCC-1.1.0-Validation-success-020321-VALIDIMG.csv',
            # there is no test
            'test': 'Validation_GCC-1.1.0-Validation-success-020321-VALIDIMG.csv'
        }

        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.metadata_dir, target_split_fp))
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        self.metadata = metadata

    def _get_video_path(self, sample):
        # conceptual captions uses this hashing to create the filename
        rel_dir = 'training'
        if self.split != 'train':
            rel_dir = 'validation'
        rel_fp = os.path.join(rel_dir, str(zlib.crc32(sample['thumbnailUrl'].encode('utf-8')) & 0xffffffff))
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample['caption']
