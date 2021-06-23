# Frozen️ in Time ❄️️️️️⏳
A Joint Video and Image Encoder for End-to-End Retrieval
----
project page | [arXiv](https://arxiv.org/abs/2104.00650) | [webvid-data](https://github.com/m-bain/webvid-dataset)
![alt text](arch.jpg)
Repository containing the code, models, data for end-to-end retrieval. WebVid data can be found [here](https://m-bain.github.io/webvid-dataset/)

----
### Preparation

1. Create conda env `conda env create -f requirements/frozen.yml`

2. Create data / experiment folders `mkdir data; mkdir exps`, note this can just be a symlink to where you want to store big data.


### Finetuning (benchmarks: MSR-VTT)

1. `wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip -d data`

2. Change `num_gpus` in the config file accordingly. 

3. Train `python train.py --config configs/msrvtt_4f_i21k.json`

4. Test `python test.py --resume exps/models/{EXP_NAME}/{EXP_TIMESTAMP}/model_best.pth`

For finetuning a pretrained model, set `"load_checkpoint": "PATH_TO_MODEL"` in the config file.

### Pretraining

1. Download WebVid-2M (see https://github.com/m-bain/webvid-dataset)

2. Download CC-3M (see https://ai.google.com/research/ConceptualCaptions/download)

3. Train. `python train.py --config CONFIG_PATH`. Here are the different options:
    
    **a. Dataset combinations**
    
        i. CC-3M + WebVid2M: configs/cc-webvid2m-pt-i2k.json
        ii. WebVid2M : configs/webvid2m-pt-i2k.json
        
    You can add in an arbitrary number of image/video datasets for pre-training by adding as many dataloaders to the config file dataloader list as your heart desires. Adding more datasets will likely to higher downstream performance. 
    
    **b. Number of frames**
    
    For image datasets, this should always be set to `video_params": {"num_frames": 1, ...}`.
    
    For video datasets, set this to what you want.
    N.B. More frames requires = more gpu memory.
    
    If, like us, you are not a big company and have limited compute, then you will benefit by training via a curriculum on the number of frames.
    A lot of the knowledge can be learned in the 1-frame setting, as we show in the paper. You can then finetune with more frames.
    
    **c. Finetuning**
    
    Set `"load_checkpoint": "FULL_MODEL_PATH"` in the config file. You can now use different experiment params, such as num_frames, to do curriculum learning for example.

### Pretrained Weights

 * [CC-3M+WebVid-2M, 4-frames, base_patch_16_224](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid2m-4f_stformer_b_16_224.pth.tar)



### Experiment Logging and Visualising
This repository uses a sacred backbone for logging and tracking experiments, with a neptune front end. It makes life a lot easier.
If you want to activate this:
1. Create a [neptune](https://neptune.ai) account.
2. Create a project, copy in your credentials in `train.py` and remove the ValueError
3. Set `neptune: true` in your config files.


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


## Acknowledgements

This code is based off the pytorch-template https://github.com/victoresque/pytorch-template

As well as many good practices adopted from Samuel Albanie's  https://github.com/albanie/collaborative-experts
