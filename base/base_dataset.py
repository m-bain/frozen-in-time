from torch.utils.data import Dataset
import random
import cv2
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class TextVideoDataset(Dataset):
    def __init__(self,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1
                 ):
        self.text_params = text_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        self.metadata_dir = os.path.expandvars(metadata_dir)
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self._load_metadata()

    def _load_metadata(self):
        pass

    def _get_video_path(self, sample):
        pass

    def _get_caption(self, sample):
        pass

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'rand'
        if self.split == 'test':
            frame_sample = 'uniform'

        try:
            imgs, idxs = self._load_frames_from_video(video_fp, self.video_params['num_frames'], frame_sample)
        except:
           if video_loading == 'strict':
               raise ValueError(f'Video loading failed for {video_fp}, video loading for this dataset is strict.')
           else:
               imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
               imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': sample.name}
        data = {'video': final, 'text': caption, 'meta': meta_arr}
        return data

    def _load_frames_from_video(self, video_path, num_frames, sample='rand', fix_start=None):
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened())
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get indexes of sampled frames
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        elif fix_start is not None:
            raise NotImplementedError
        else:
            raise NotImplementedError

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f'images/{index}.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                raise ValueError

        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs
