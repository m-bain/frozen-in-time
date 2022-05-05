import json
import os
import random

import numpy as np
import pandas as pd
import glob

from base.base_dataset import TextVideoDataset

class VideoDirectory(TextVideoDataset):
    def _load_metadata(self):

        if self.split != 'test':
            raise NotImplementedError("Assumes inference, no text, hence cant be used for training...")
        ## assumes mkv
        TARGET_EXT = "*.mp4"
        video_glob = os.path.join(self.data_dir, '**', target_ext)
        video_li = glob.glob(video_glob, recursive=True)
        video_li = [x.replace(self.data_dir, '').strip('/') for x in video_li]
        video_li = sorted(video_li)
        self.metadata = pd.Series(video_li)

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, sample), sample

    def _get_caption(self, sample):
        return 'placeholder'


class CMDShotFeats(VideoDirectory):
    def _load_metadata(self):
        super()._load_metadata()

        #ftrs_dir = "/scratch/shared/beegfs/maxbain/datasets/CondensedMoviesShots/features/CC-WebVid2M-4f-pt1f/0522_143949"
        ftrs_dir = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/features/CLIP/clip-vit-base-patch16"
        print("### WARNING ### :: using : ", ftrs_dir)
        csv_files = glob.glob(os.path.join(ftrs_dir, "ids_test_*.csv"))

        dfs = []
        for csv_fp in csv_files:
            dfs.append(pd.read_csv(csv_fp))
        if len(dfs) > 0:
            dfs = pd.concat(dfs)
            self.metadata = self.metadata[~self.metadata.isin(dfs['0'])]
        self.metadata = self.metadata.reset_index()
        self.metadata = self.metadata[0]
        self.metadata = pd.DataFrame({"0": self.metadata})
        print(self.metadata)
        print(len(self.metadata), " to do...")

if __name__ == "__main__":
    from data_loader import transforms
    tsfms = transforms.init_transform_dict()
    ds = CMDShotFeats(
        'CMDShotFeats',
        {"input": "text"},
        {"input_res": 224, "num_frames": 4},
        data_dir='/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/videos',
        split='test',
        tsfms=tsfms['test']
    )

    for x in range(100):
        print(ds.__getitem__(x))
    #random_set = ds.metadata.sample(100)
    #random_set['captions'] = random_set['captions'].str[0]