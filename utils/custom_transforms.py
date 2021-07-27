import numbers
from typing import List, Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional_pil as F_pil, functional_tensor as F_t
from torchvision.transforms.functional import center_crop, crop


def _get_image_size(img: Tensor) -> List[int]:
    """Returns image size as [w, h]
    """
    if isinstance(img, torch.Tensor):
        return F_t._get_image_size(img)

    return F_pil._get_image_size(img)

def center_plus_four_crops(img: Tensor, size: List[int],
                              margin_h: int, margin_w: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Crop the given image into four tiled borders and the central crop.
    """

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = _get_image_size(img)

    crop_height, crop_width = size

    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    if crop_width + margin_w > image_width:
        msg = "Requested margin size {} + input {} is bigger than input size {}"
        raise ValueError(msg.format((margin_h, margin_w), size, (image_height, image_width)))

    #vertical_border_height = image_height - crop_height
    #horizontal_border_height = image_width - crop_width

    #x1 = horizontal_border_height // 2
    x11 = (image_width - crop_width - 2 * margin_w) // 2
    x12 = x11 + margin_w
    x21 = x12 + crop_width
    x22 = x21 + margin_w

    y11 = (image_height - crop_height - 2 * margin_h) // 2
    y12 = y11 + margin_h
    y21 = y12 + crop_height
    y22 = y21 + margin_h

    tl = crop(img, y11, x11, margin_h, margin_w + crop_width)
    tr = crop(img, y11, x21, margin_h + crop_height, margin_w)
    bl = crop(img, y12, x11, margin_h + crop_height, margin_w)
    br = crop(img, y21, x12, margin_h, margin_w + crop_width)
    center = center_crop(img, [crop_height, crop_width])

    return tl, tr, bl, br, center



def center_plus_twohori_crops(img: Tensor, size: List[int],
                              margin_w: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Crop the given image into four tiled borders and the central crop.
    """

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = _get_image_size(img)

    crop_height, crop_width = size

    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    if crop_width + margin_w > image_width :
        msg = "Requested margin size {} + input {} is bigger than input size {}"
        raise ValueError(msg.format((0, margin_w), size, (image_height, image_width)))

    # vertical_border_height = image_height - crop_height
    # horizontal_border_height = image_width - crop_width

    # x1 = horizontal_border_height // 2
    x11 = (image_width - crop_width - 2 * margin_w) // 2
    x12 = x11 + margin_w
    x21 = x12 + crop_width

    y11 = (image_height - crop_height) // 2

    left = crop(img, y11, x11, crop_height, margin_w)
    right = crop(img, y11, x21, crop_height, margin_w)
    center = center_crop(img, [crop_height, crop_width])

    return left, right, center

from torch import nn
class TwoHoriCrop(nn.Module):
    def __init__(self, size, margin_w):
        super().__init__()
        self.size = size
        self.margin_w = margin_w

    def forward(self, x):
        return center_plus_twohori_crops(x, self.size, self.margin_w)

if __name__ == "__main__":
    from PIL import Image

    img = Image.open('visualisations/guitar.png')
    crops = center_plus_four_crops(img, [336, 336], 112, 112)
    order = ['tl', 'tr', 'bl', 'br', 'center']

    for idx, subimg in zip(order, crops):
        subimg.save(f'visualisations/guitar_{idx}.png')

    crops = center_plus_twohori_crops(img, [448, 448], 112)
    order = ['left', 'right', 'center2']

    for idx, subimg in zip(order, crops):
        subimg.save(f'visualisations/guitar_{idx}.png')
