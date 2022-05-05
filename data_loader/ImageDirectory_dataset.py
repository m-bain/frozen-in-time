import json
import os
import random

import numpy as np
import pandas as pd
import glob

from base.base_dataset import TextImageDataset


class ImageDirectory(TextImageDataset):
    def _load_metadata(self):
        if self.split != 'test':
            raise NotImplementedError("Assumes inference, no text, hence cant be used for training...")

        TARGET_EXT = "*.jpg"
        img_glob = os.path.join(self.data_dir, "**", TARGET_EXT)
        img_li = glob.glob(img_glob, recursive=True)
        img_li = [x.replace(self.data_dir, '').strip('/') for x in img_li]
        img_li = sorted(img_li)
        self.metadata = pd.Series(img_li)

    def _get_video_path(self, sample):
        # return full filepath, and relative filepath
        return os.path.join(self.data_dir, sample), sample

    def _get_caption(self, sample):
        return 'placeholder'


if __name__ == "__main__":
    from data_loader import transforms
    tsfms = transforms.init_transform_dict()
    ds = ImageDirectory(
        'DummyImg',
        {"input": "text"},
        {"input_res": 224, "num_frames": 1},
        data_dir='/users/maxbain/Desktop/test',
        split='test',
        tsfms=tsfms['test']
    )

    for x in range(100):
        print(ds.__getitem__(x))
    #random_set = ds.metadata.sample(100)
    #random_set['captions'] = random_set['captions'].str[0]