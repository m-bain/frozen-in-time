import argparse
import torch
from tqdm import tqdm
import data_loader.data_loader as module_data
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.model import sim_matrix
import pandas as pd
import numpy as np
from sacred import Experiment
import transformers
from utils.util import state_dict_data_parallel_fix
from trainer.trainer import verbose
ex = Experiment('test')

@ex.main
def run():

    # setup data_loader instances
    config._config['data_loader']['args']['split'] = 'test'
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['sliding_window_stride'] = config._config['sliding_window_stride']
    data_loader = config.initialize('data_loader', module_data)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    #logger.info('Loading checkpoint: {} ...'.format(config.resume))

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

    meta_arr = []
    text_embed_arr = []
    vid_embed_arr = []
    print(len(data_loader))
    with torch.no_grad():
        for i, data in tqdm(tqdm(enumerate(data_loader))):
            # leave this for now since not doing anything on the gpu
            meta_arr.append(data['meta'])
            if tokenizer is not None:
                data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}
            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)

            text_embed, vid_embed = model(data, return_embeds=True)
            text_embed_arr.append(text_embed)
            vid_embed_arr.append(vid_embed)

    text_embeds = torch.cat(text_embed_arr)
    vid_embeds = torch.cat(vid_embed_arr)

    mask = None
    if data_loader.dataset.sliding_window_stride != -1:
        cpu_vid_embeds = vid_embeds.cpu().detach()
        cpu_text_embeds = text_embeds.cpu().detach()

        li_vid_embeds = [x for x in cpu_vid_embeds]
        li_txt_embeds = [x for x in cpu_text_embeds]
        videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
        raw_caps = pd.Series([x['raw_captions']] for x in meta_arr).explode().explode()
        vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                               'captions': raw_caps})
        new_vid_embeds = []
        new_txt_embeds = []
        for vid in vid_df['videoid'].unique():
            tdf = vid_df[vid_df['videoid'] == vid]
            tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
            tvembeds = tvembeds.mean(dim=0)
            new_vid_embeds.append(tvembeds)

            for cap in tdf['captions'].unique():
                cdf = vid_df[vid_df['captions'] == cap]
                ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                new_txt_embeds.append(ttembeds[0])

        vid_embeds = torch.stack(new_vid_embeds).cuda()
        text_embeds = torch.stack(new_txt_embeds).cuda()

    sims = sim_matrix(text_embeds, vid_embeds)
    sims = sims.detach().cpu().numpy()

    nested_metrics = {}
    for metric in metric_fns:
        metric_name = metric.__name__
        res = metric(sims, query_masks=mask)
        verbose(epoch=0, metrics=res, name="", mode=metric_name)
        nested_metrics[metric_name] = res

    #if config.config['visualizer']:
    #    raise NotImplementedError

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

    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()
