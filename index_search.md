Extracting features & creating a faiss index

1. `conda env create`
2. `conda install -c pytorch faiss-cpu`
3. To extract features for a directory containing images (as well as subdirectory of images):
   - Edit the field `[dataloader][args][data_dir]` field in the config file `./configs/image_feat_extract.json` or `./configs/video_feat_extract.json` for image or video data respectively.
   N.B. the code assumes all images,videos are .jpg,.mp4, if this is not the case you can edit the file `data_loader/ImageDirectory_dataset.py`
   - Edit the field `"load_checkpoint"` to the path of the downloaded pretrained model. 
4. then run feature extraction `python test.py --config configs/image_feat_extract.json --save_feats SAVE_FTR_DIR --save_type video`. Assumes there is a gpu available.

csv file(s) and numpy array(s) are saved to SAVE_FTR_PATH. When there's more than 1000 images/videos it creates multiple files to avoid OOM. In this case we need aggregate these into a single numpy array and single csv file:

4. `python scripts/agg_ids_embeds.py --path_to_saved_features SAVE_FTR_DIR`

Next, create a FAISS index from the single numpy array and csv file. If your images/videos have unique integer IDs then we can create and index with ids (so you dont need to carry around the csv mapping, use the `--add_index_with_ids` in this case):

5. `python scripts/create_faiss_index.py --path_to_saved_features SAVE_FTR_DIR`