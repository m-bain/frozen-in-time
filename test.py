import argparse

import pandas as pd
import torch
import transformers
from sacred import Experiment
from tqdm import tqdm
import glob
import data_loader.data_loader as module_data
import model.metric as module_metric
import model.model as module_arch
from model.model import compute_similarity
from parse_config import ConfigParser
from trainer.trainer import verbose
from utils.util import state_dict_data_parallel_fix
import numpy as np
import os
from utils.visualisation import visualise_text_video_sim
import copy

ex = Experiment('test')


@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride
    # config._config['data_loader']['args']['video_params']['num_frames'] = 120

    data_loader = config.initialize('data_loader', module_data)
    n_samples = len(data_loader.dataset)

    text_model_name = config['arch']['args']['text_params']['model']
    if "openai/clip" in text_model_name:
        tokenizer_builder = transformers.CLIPTokenizer
    else:
        tokenizer_builder = transformers.AutoTokenizer
    tokenizer = tokenizer_builder.from_pretrained(
        text_model_name,
        model_max_length=config['arch']['args']['text_params'].get('max_length', 1e6),
        TOKENIZERS_PARALLELISM=False)

    # build model architecture
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    ctr = 0
    save_part = None
    if args.save_feats:
        part_seq = [int(x.split('_')[-1].split('.')[0]) for x in
                    glob.glob(os.path.join(args.save_feats, "ids_test_*.csv"))]
        if len(part_seq) > 0:
            save_part = max() + 1
        else:
            save_part = 0
        print(F"##### WARNING SAVE_PART STARTING AT {save_part}, MAKE SURE THIS IS THE NEWEST")

    meta_arr = []
    text_embed_arr = []
    vid_embed_arr = []
    text_mask_arr = []
    vid_mask_arr = []
    print(len(data_loader))
    with torch.no_grad():
        for i, data_og in tqdm(tqdm(enumerate(data_loader))):
            # leave this for now since not doing anything on the gpu
            data = copy.deepcopy(data_og)
            del data_og
            if tokenizer is not None:
                if args.vis_token_similarity:
                    data['meta']['tokenized_captions'] = [tokenizer.tokenize(x) for x in data['text']]
                data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}
            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)

            output = model(data)
            text_embed_arr.append(output['text'].cpu().detach())
            vid_embed_arr.append(output['video'].cpu().detach())

            # meta stuff
            meta_arr.append(data['meta'])
            text_mask_arr.append(data['text']['attention_mask'].cpu())
            vid_mask_arr.append(data['video_mask'])

            ctr += len(data['video'])
            # save every 1mil samples to avoid OOM
            if args.save_feats is not None and ctr > 1e4:
                ctr = 0
                text_embeds = torch.cat(text_embed_arr)
                vid_embeds = torch.cat(vid_embed_arr)
                meta_arr_cat = {key: [] for key in meta_arr[0].keys()}
                for meta in meta_arr:
                    for key, val in meta.items():
                        meta_arr_cat[key].append(val)
                meta_arr = meta_arr_cat
                for key, val in meta_arr.items():
                    if isinstance(val[0], list):
                        val = [item for sublist in val for item in sublist]
                        meta_arr[key] = val
                    elif isinstance(val[0], torch.Tensor):
                        meta_arr[key] = torch.cat(val)
                    else:
                        raise NotImplementedError
                save_feats(vid_embeds, text_embeds, meta_arr, args.save_feats, args.save_type,
                           data_loader.dataset.split, save_part=save_part)
                text_embed_arr = []
                vid_embed_arr = []
                meta_arr = []
                save_part += 1

    vid_embeds = torch.cat(vid_embed_arr)
    vid_masks = torch.cat(vid_mask_arr)

    meta_arr_cat = {key: [] for key in meta_arr[0].keys()}
    for meta in meta_arr:
        for key, val in meta.items():
            meta_arr_cat[key].append(val)
    meta_arr = meta_arr_cat
    for key, val in meta_arr.items():
        if isinstance(val[0], list):
            val = [item for sublist in val for item in sublist]
            meta_arr[key] = val
        elif isinstance(val[0], torch.Tensor):
            meta_arr[key] = torch.cat(val)
        else:
            raise NotImplementedError

    similarity_type = config._config['loss']['args'].get("similarity", "single")
    from torch import nn
    max_txt_len = max([x.shape[1] for x in text_mask_arr])
    for idx, txt_mask in enumerate(text_mask_arr):
        to_pad = max_txt_len - txt_mask.shape[1]
        # move tokens to last dim, pad, then move tokens back to mid dim
        text_mask_arr[idx] = nn.functional.pad(txt_mask, (0, to_pad))
    text_embeds = torch.cat(text_embed_arr)
    text_masks = torch.cat(text_mask_arr)
    mask = None
    if data_loader.dataset.sliding_window_stride != -1:
        cpu_vid_embeds = vid_embeds
        cpu_text_embeds = text_embeds

        li_vid_embeds = [x for x in cpu_vid_embeds]
        li_txt_embeds = [x for x in cpu_text_embeds]
        li_vid_masks = [x for x in vid_masks]
        li_txt_masks = [x for x in text_masks]
        videoids = pd.Series(meta_arr['paths'])
        raw_caps = pd.Series(meta_arr['raw_captions'])
        vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                               'captions': raw_caps, 'vid_mask': li_vid_masks, 'txt_mask': li_txt_masks})
        new_vid_embeds = []
        new_txt_embeds = []
        new_vid_masks = []
        new_txt_masks = []
        for vid in vid_df['videoid'].unique():
            tdf = vid_df[vid_df['videoid'] == vid]
            tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
            tvembeds = tvembeds.mean(dim=0)
            new_vid_embeds.append(tvembeds)
            tvmasks = torch.stack(tdf['vid_mask'].values.tolist())
            new_vid_masks.append(tvmasks.max(dim=0)[0])

            for cap in tdf['captions'].unique():
                cdf = vid_df[vid_df['captions'] == cap]
                ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                new_txt_embeds.append(ttembeds[0])
                ttmasks = torch.stack(cdf['txt_mask'].values.tolist())
                new_txt_masks.append(ttmasks[0])

        vid_embeds = torch.stack(new_vid_embeds)
        text_embeds = torch.stack(new_txt_embeds)
        vid_masks = torch.stack(new_vid_masks)
        text_masks = torch.stack(new_txt_masks)

    if args.split != 'train':  # because train is usually too big
        chunk = True
        if not chunk:
            sims, _ = compute_similarity(text_embeds, vid_embeds, text_masks, vid_masks, style=similarity_type,
                                         return_raw=args.vis_token_similarity)
            if args.vis_token_similarity:
                mask = _
        else:
            chunk_size = 100
            sim_row_arr = []
            for tdx in range(0, len(text_embeds), chunk_size):
                print(tdx, ' / ', len(text_embeds), ' ...')
                t_embed = text_embeds[tdx:tdx + chunk_size]
                t_mask = text_masks[tdx:tdx + chunk_size]
                sim_row = []
                for vdx in range(0, len(vid_embeds), chunk_size):
                    v_embed = vid_embeds[vdx:vdx + chunk_size]
                    v_mask = vid_masks[vdx:vdx + chunk_size]
                    sim_chunk, _ = compute_similarity(t_embed, v_embed, t_mask, v_mask, style=similarity_type,
                                                      return_raw=args.vis_token_similarity)
                    sim_row.append(sim_chunk)
                sim_row = torch.cat(sim_row, dim=1)
                sim_row_arr.append(sim_row)

            sims = torch.cat(sim_row_arr, dim=0)

        sims = sims.numpy()

        # if not args.vis_token_similarity:
        nested_metrics = {}
        for metric in metric_fns:
            metric_name = metric.__name__
            res = metric(sims, query_masks=mask)
            verbose(epoch=0, metrics=res, name="", mode=metric_name)
            nested_metrics[metric_name] = res
        # else:
        #     visualise_text_video_sim(sims, mask, meta_arr, num_vis=10)

    # if config.config['visualizer']:
    #    raise NotImplementedError
    if args.save_feats is not None:
        if save_part == 0:
            save_part = None

        save_feats(vid_embeds, text_embeds, meta_arr, args.save_feats, args.save_type, data_loader.dataset.split,
                   save_part=save_part)

        # meta_arr['frame_id'] = meta_arr['frame_id'].numpy()
        if save_part is None:
            fn = f'meta_arr.npy'
        else:
            fn = f'meta_arr_{save_part}.npy'
        np.save(os.path.join(args.save_feats, fn), meta_arr)


def save_feats(vid_embeds, text_embeds, meta_arr, save_feats, save_type, split, save_part=None):
    vid_embeds = vid_embeds.cpu().detach().numpy()
    text_embeds = text_embeds.cpu().detach().numpy()

    if save_part is not None:
        vid_fn = f'vid_embeds_{split}_{save_part}.npy'
        txt_fn = f'txt_embeds_{split}_{save_part}.npy'
        csv_fn = f'ids_{split}_{save_part}.csv'
    else:
        vid_fn = f'vid_embeds_{split}.npy'
        txt_fn = f'txt_embeds_{split}.npy'
        csv_fn = f'ids_{split}.csv'

    vid_embeds_save_fp = os.path.join(save_feats, vid_fn)
    txt_embeds_save_fp = os.path.join(save_feats, txt_fn)

    if save_type in ['video', 'both']:
        np.save(vid_embeds_save_fp, vid_embeds)
    if save_type in ['text', 'both']:
        np.save(txt_embeds_save_fp, text_embeds)

    videoids = pd.Series(meta_arr['paths'])
    # frame_ids = pd.Series(meta_arr['frame_id'].numpy())
    meta_df = pd.DataFrame({'0': videoids})
    meta_df.to_csv(os.path.join(save_feats, csv_fn), index=False)

    if len(videoids) != len(vid_embeds):
        import pdb;
        pdb.set_trace()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--save_type', default='both', choices=['both', 'text', 'video'],
                      help='Whether to save video, text or both feats. If running on inference videos, text is just a placeholder')
    args.add_argument('--vis_token_similarity', action='store_true')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=16, type=int,
                      help='size of batch')
    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()
