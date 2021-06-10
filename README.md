# Frozen️ in Time ❄️️️️️⏳
### A Joint Video and Image Encoder for End-to-End Retrieval
([arXiv](https://arxiv.org/abs/2104.00650))
----
Repository to contain the code, models, data for end-to-end retrieval.

### Work in progress ###

Code provided to train end-to-end model on MSRVTT.

1. Create conda env `conda env create -f requirements/frozen.yml`

2. Download MSRVTT data `mkdir data; mkdir exps; wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip -d data`

3. Change `num_gpus` in the config file accordingly. 

4. Train `python train.py --config configs/msrvtt_4f_i21k.json`

5. Test `python test.py --resume exps/models/{EXP_NAME}/{EXP_TIMESTAMP}/model_best.pth`

## Cite

If you use this code in your research, please cite:

<div class="highlight highlight-source-shell"><pre>
@misc{bain2021frozen,
      title={Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval}, 
      author={Max Bain and Arsha Nagrani and Gül Varol and Andrew Zisserman},
      year={2021},
      eprint={2104.00650},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
</pre></div>

TODO:

[x] conda env

[x] msrvtt data zip

[ ] pretrained models

[ ] webvid data

[ ] Other benchmarks 

![alt text](arch.jpg)