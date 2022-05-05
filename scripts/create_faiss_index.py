import argparse
import faiss
from pathlib import Path
import numpy as np
import os

def create_index(
        index: str,
        embed_dim: int,
        nlist: int,
):
    index = faiss.index_factory(embed_dim, f"{index}{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
    return index


def load_feats(
        feat_fp: Path,
        normalize: bool = True
):
    feats = np.load(feat_fp)
    if normalize:
        faiss.normalize_L2(feats)
    return feats



def main(args):
    db_dir = Path(args.path_to_saved_features)
    feats_db_fn = f"{args.data_type}_embeds_{args.data_split}.npy"
    feats_db_fp = db_dir / feats_db_fn

    #feats_db = load_feats(feats_db_fp, normalize=True)
    feats_db = np.random.rand(50000, 256).astype(np.float32)
    print(f"Training index...")
    feat_train_idx = np.random.choice(len(feats_db), int(len(feats_db)*args.train_index_frac))
    feats_train = feats_db[feat_train_idx]

    nlist = args.nlist
    if nlist is None:
        db_len = len(feats_db)
        nlist = int(4 * (db_len ** 0.5))
    index = create_index(args.index, args.embed_dim, nlist)
    index.train(feats_train)

    # add to...
    ids_db_fn = f"ids_{args.data_split}.csv"

    print(f"Adding {feats_db_fn} to index...")
    feats_db_fp = db_dir / feats_db_fn

    feats_db = load_feats(feats_db_fp, normalize=True)


    print(f"... adding database of size {index.ntotal} to index.")
    #### Optionally use an index with ID's, requires int, so need some mapping. e.g. assume image filename is unique int
    if args.add_index_with_ids:
        ids_db_fp = db_dir / ids_db_fn
        ids_db = pd.read_csv(ids_db_fp)
        ids_db['videoid'] = ids_db['0'].str.split('/').str[-1].str.split('.').str[0]
        ids_db.set_index('videoid', inplace=True)
        ids_db.index = ids_db.index.astype(int)
        ids = ids_db.index.values
        index.add_with_ids(feats_db, ids)
    else:
        index.add(feats_db)

    index_fp = os.path.join(args.path_to_saved_features, f"{args.index}_{args.data_split}.index")
    print(f"Writing index to ...\n{index_fp}")

    faiss.write_index(index, index_fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_saved_features', required=True, type=str)
    parser.add_argument('--data_type', default='vid')
    parser.add_argument('--data_split', default='test')
    parser.add_argument('--train_index_frac', default=0.1)
    parser.add_argument('--nlist', default=None, type=int,
                        help='Number of centroids to use')
    parser.add_argument('--embed_dim', default=256, type=int,
                        help='dimension size of embeddings')
    parser.add_argument('--index', default='IVF', choices=['IVF'],
                        help='Choice of FAISS Index.')
    parser.add_argument('--add_index_with_ids', action='store_true')


    args = parser.parse_args()
    main(args)
