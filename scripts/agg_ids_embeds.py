import numpy as np
import pandas as pd
import glob
import os
from ast import literal_eval
import argparse


# dir = '/scratch/shared/beegfs/maxbain/datasets/CondensedMoviesShots/features/CC-WebVid2M-4f-pt1f/0522_143949'
# dir = '/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/features/CLIP4CLIP/cmd_batch_size_test__multirun_2021-12-02_03-35-39__28'
# dir = "/work/maxbain/Libs/long-video-modelling/corpus/feats/CLIP4CLIP_msr/msr_mean_2022-01-31_17-54-49"
def main(args):
    dir = args.path_to_saved_features
    split = args.data_split
    dtype = args.data_type
    id_files = glob.glob(os.path.join(dir, f'ids_{split}*.csv'))
    id_fns = [x.split('/')[-1].split('.csv')[0] for x in id_files]

    embed_arr = []
    df_arr = []
    for idfn in id_fns:
        print(idfn)
        if idfn != f"ids_{split}":
            qdf = pd.read_csv(os.path.join(dir, idfn + '.csv'))
            embed_fn = f"{dtype}_embeds_{split}_{idfn.split('_')[-1]}.npy"
            queries_full_fp = os.path.join(dir, embed_fn)
            embeds = np.load(queries_full_fp)
            if len(embeds) != len(qdf):
                print("not same")
                import pdb;
                pdb.set_trace()
            else:
                if args.frame_info:
                    qdf['1'] = qdf['1'].apply(lambda x: [y.strip().strip(']').strip('[') for y in x.split(' ') if y != ''])
                    qdf['1'] = qdf['1'].apply(lambda x: [int(y) for y in x if y != ''])
                    qdf = qdf.explode('1')
                    qdf.reset_index(inplace=True)
                    qdf = qdf[qdf['1'] != -1]
                    qdf['1'] = '[' + qdf['1'].astype(str) + '.]'
                embeds = embeds.reshape(-1, embeds.shape[-1])
                embeds = embeds[qdf.index]
                df_arr.append(qdf)
                embed_arr.append(embeds)

    embed_arr = np.concatenate(embed_arr, axis=0)
    df_arr = pd.concat(df_arr)
    df_arr.to_csv(os.path.join(dir, f'{dtype}_ids_{split}.csv'), index=False)
    np.save(os.path.join(dir, f'{dtype}_embeds_{split}.npy'), embed_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_saved_features', required=True, type=str)
    parser.add_argument('--data_type', default='vid')
    parser.add_argument('--data_split', default='test')
    parser.add_argument('--frame_info', default=False)
    args = parser.parse_args()
    main(args)
