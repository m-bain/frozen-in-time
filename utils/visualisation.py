import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')


def visualise_path(pred, target, window):
    """
    :param pred: (P, 2) Tensor where P is the number of predictions, and 2 is the (i,j) coordinate
    :param target: (T, 2) Tensor where T is the number of targets, and 2 is the (i,j) coordinate
    :param dims: (H, W) tup/list the desired height and width of matrix (should be >= to max(i), max(j))
    :param assignment_method: Method of assignment (dtw, minimum etc.)
    :return: image, visualisation of path prediction and target.
    """
    tp = torch.Tensor((64, 191, 64))
    fp = torch.Tensor((191, 64, 64))
    gt = torch.Tensor((102, 153, 255))

    grid = torch.ones_like(window).unsqueeze(0).repeat(3, 1, 1) * 255
    inf = 130 * torch.ones_like(grid)
    grid = torch.where(torch.isnan(window), inf, grid)

    clip_idxs = [t[0] for t in target]
    local_idxs = np.unique(np.array(clip_idxs)).tolist()

    for t in target:
        local_idx = local_idxs.index(t[0])
        grid[:, local_idx, t[1]] = gt

    for p in pred:
        local_idx = local_idxs.index(p[0])
        if (grid[:, local_idx, p[1]] == gt).all():
            grid[:, local_idx, p[1]] = tp
        else:
            grid[:, local_idx, p[1]] = fp

    return grid / 255


def batch_path_vis(pred_dict, target, window):
    grids = []

    window = window.cpu()
    for key, pred in pred_dict.items():
        tmp_window = window
        if key == 'min_dist':
            tmp_window = torch.zeros_like(window)
        grids.append(visualise_path(pred, target, tmp_window))

    return torch.stack(grids)


if __name__ == "__main__":
    pred = [[1, 1], [2, 4]]
    gt = [[1, 1], [3, 4]]
    window = torch.zeros((5, 6))
    visualise_path(pred, gt, window)
