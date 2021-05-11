from base import BaseDataLoader, BaseDataLoaderExplicitSplit
from base import TextVideoDataset
from data_loader.MSRVTT_dataset import MSRVTT
from torchvision import transforms


def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1):
    kwargs = dict(
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample
    )

    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):

        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, split, tsfm, cut, subsample)

        if split != 'train':
            shuffle = False

        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name


def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ])
    }
    return tsfm_dict
