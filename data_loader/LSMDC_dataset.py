import os

import numpy as np
import pandas as pd

from base.base_dataset import TextVideoDataset


class LSMDC(TextVideoDataset):
    def _load_metadata(self):
        split_paths = {key: os.path.join(self.metadata_dir, 'structured-symlinks', f'{key}_list.txt') for key in
                       ['train', 'val', 'test']}
        df_dict = {key: pd.read_csv(val, names=['videoid']) for key, val in split_paths.items()}
        #### subsample_val

        self.split_sizes = {key: len(val) for key, val in df_dict.items()}
        target_vids = df_dict[self.split]
        # target_vids = target_vids['videoid'].str.split('.').str[0]
        if self.subsample < 1:
            target_vids = target_vids.sample(frac=self.subsample)
        captions = np.load(os.path.join(self.metadata_dir, 'structured-symlinks', 'raw-captions.pkl'),
                           allow_pickle=True)
        captions = pd.DataFrame.from_dict(captions, orient='index')
        captions['captions'] = captions.values.tolist()
        target_vids.set_index('videoid', inplace=True)
        target_vids['captions'] = captions['captions']
        # import pdb; -.set_trace()
        # captions = captions[captions.index.isin(target_vids.str['videoid'].split('.').str[0])]
        self.metadata = target_vids
        frame_tar_list = pd.read_csv(os.path.join(self.metadata_dir, 'frame_tar_list.txt'), names=['fp'])

        frame_tar_list['fn'] = frame_tar_list['fp'].str.split('/').str[-2:].str.join('/')
        frame_tar_list['fn'] = frame_tar_list['fn'].str.replace('.tar', '')
        frame_tar_list['vid_stem'] = frame_tar_list['fn'].str.split('/').str[-1]

        frame_tar_list = frame_tar_list[frame_tar_list['vid_stem'].isin(self.metadata.index)]

        frame_tar_list.set_index('vid_stem', inplace=True)
        self.metadata['fn'] = frame_tar_list['fn']
        self.metadata['captions'] = self.metadata['captions'].apply(lambda x: [ii for ii in x if ii is not None])
        self.metadata['num_captions'] = self.metadata['captions'].str.len()
        self.metadata['captions'] = self.metadata['captions'].apply(lambda x: [' '.join(ii) for ii in x])

        if 'videoid' not in self.metadata.columns:
            self.metadata['videoid'] = self.metadata.index


    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', sample['fn'] + '.avi'), sample.name + '.avi'

    def _get_caption(self, sample):
        if len(sample['captions']) != 1:
            raise NotImplementedError
        return sample['captions'][0]
