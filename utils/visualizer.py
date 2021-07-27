"""A simple HTML visualizer.

It is based on the Cycle-GAN codebase:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
import os
from pathlib import Path

import numpy as np

from . import html


class RetrievalVis:
    """This class includes several functions that can display/save images.

    It uses a Python library 'visdom' for display, and a Python library 'dominate'
    (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, exp_name, web_dir, src_video_dir, vis_vid_freq, num_samples=50):
        """Initialize the Visualizer class
        Create an HTML object for saveing HTML filters
        """
        self.name = exp_name
        self.web_dir = web_dir
        self.vis_vid_freq = vis_vid_freq
        self.img_dir = os.path.join(self.web_dir, "images")
        self.num_samples = num_samples

        self.data_type = 'images' # 'images' or 'videos'
        assert self.data_type in ('images', 'videos')

        print(f"create web directory {self.web_dir}...")
        mkdirs([self.web_dir, self.img_dir])

        # cluster specific
        if "$TMPDIR" in src_video_dir:
            src_video_dir = src_video_dir.replace("$TMPDIR", os.environ['TMPDIR'])

        src_dir = Path(src_video_dir).absolute()
        print(f"symlinking videos from {src_dir}...")
        sym_dir = (Path(self.web_dir) / "videos").absolute()
        if sym_dir.is_symlink():
            os.remove(sym_dir)
        sym_dir.symlink_to(src_dir)

    def visualize_ranking(self, sims, epoch, meta, nested_metrics):
        if not (self.vis_vid_freq and epoch % self.vis_vid_freq == 0):
            return

        dists = -sims
        np.random.seed(0)
        sorted_ranks = np.argsort(dists, axis=1)
        gt_dists = np.diag(dists)
        rankings = []
        vis_top_k = 5
        hide_gt = False
        # num_indep_samples = 1
        # random_seeds = np.arange(num_indep_samples)
        sample = np.random.choice(np.arange(dists.shape[0]), size=self.num_samples,
                                  replace=False)
        for ii in sample:
            ranked_idx = sorted_ranks[ii][:vis_top_k]
            gt_captions = meta["raw_captions"][ii]
            # if args.sample_single_gt_caption:
            #     gt_captions = np.random.choice(gt_captions, 1).tolist()
            datum = {
                "gt-sim": -gt_dists[ii],
                "gt-captions": gt_captions,
                "gt-rank": np.where(sorted_ranks[ii] == ii)[0][0],
                "gt-path": meta["paths"][ii],
                "top-k-sims": -dists[ii][ranked_idx],
                "top-k-paths": np.array(meta["paths"])[ranked_idx],
                "hide-gt": hide_gt,
            }
            rankings.append(datum)
        self.display_current_results(
            rankings,
            epoch=epoch,
            metrics=nested_metrics["t2v_metrics"],
        )

    def display_current_results(self, rankings, epoch, metrics):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if not Path(self.web_dir).exists():
            Path(self.web_dir).mkdir(exist_ok=True, parents=True)
        print(f"updating webpage at {self.web_dir}")
        title = f"Experiment name = {self.name}"
        refresh = True
        if not refresh:
            print("DISABLING WEB PAGE REFRESH")
        webpage = html.HTML(web_dir=self.web_dir, title=title, refresh=refresh)

        msg = f"epoch [{epoch}] - {self.name}"
        webpage.add_header(msg)
        msg = (f"R1: {metrics['R1']:.1f}, "
               f"R5: {metrics['R5']:.1f}, "
               f"R10: {metrics['R10']:.1f}, "
               f"MedR: {metrics['MedR']}")
        webpage.add_header(msg)
        print(f"Top {len(rankings[0])} retreived videos at epoch: {epoch}")

        for ranking in rankings:
            vids, txts, links = [], [], []
            gt_vid_path = os.path.join('videos', ranking["gt-path"])
            #gt_captions = [" ".join(x) for x in ranking["gt-captions"]]
            gt_captions = ranking['gt-captions']
            gt_captions = "<br>" + (gt_captions) + "<br>"
            if ranking["hide-gt"]:
                txts.append(gt_captions)
                links.append("hidden")
                vids.append("hidden")
            else:
                txt = (f"{gt_captions}<br><b>Rank: {ranking['gt-rank']}, "
                       f"Sim: {ranking['gt-sim']:.3f} [{Path(ranking['gt-path']).stem}]")
                txts.append(txt)
                links.append(gt_vid_path)
                vids.append(gt_vid_path)

            for idx, (vid_path, sim) in enumerate(zip(ranking["top-k-paths"],
                                                  ranking["top-k-sims"])):
                vid_path = Path(os.path.join('videos', vid_path))
                if ranking["hide-gt"]:
                    txt = f"choice: {idx}"
                else:
                    txt = f"<b>Rank: {idx}, Sim: {sim:.3f}, [{Path(vid_path).stem}]"
                txts.append(txt)
                vids.append(vid_path)
                links.append(vid_path)
            if self.data_type == 'videos':
                webpage.add_videos(vids, txts, links, width=200)
            elif self.data_type == 'images':
                webpage.add_images(vids, txts, links, width=200)
        print(f"added {len(vids)} videos")
        webpage.save()

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
