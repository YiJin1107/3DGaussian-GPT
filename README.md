## MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers

<hr/>

[**arXiv**](http://arxiv.org/abs/2311.15475) | [**Video**](https://www.youtube.com/watch?v=UV90O1_69_o) | [**Project Page**](https://nihalsid.github.io/mesh-gpt/) <br/>


This repository contains the implementation for the paper:

[**MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers**](http://arxiv.org/abs/2311.15475) by Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, Matthias Nießner.

<div>
<div style="text-align: center">
  <img src="https://private-user-images.githubusercontent.com/932110/313438174-05cc7c73-53c7-4d8c-9514-bd2f8a7d7ed0.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA2MzcxMjIsIm5iZiI6MTcxMDYzNjgyMiwicGF0aCI6Ii85MzIxMTAvMzEzNDM4MTc0LTA1Y2M3YzczLTUzYzctNGQ4Yy05NTE0LWJkMmY4YTdkN2VkMC5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMxN1QwMDUzNDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04YjZiYTgyMTU5NzM3MTk4YTYyNTc1Njk2Y2UxZWJjZTRjODkzYzViNDFlMmExYjkzMDBhOTU4YWJmZDJlZTJkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.2uVUaTidnV3_b1V_WfTbsdwZXGzUr3otFp3wqR6tSvI" alt="animated" />
</div>
<div style="margin-top: 5px;">
MeshGPT creates triangle meshes by autoregressively sampling from a transformer model that has been trained to produce tokens from a learned geometric vocabulary. These tokens can then be decoded into the faces of a triangle mesh. Our method generates clean, coherent, and compact meshes, characterized by sharp edges and high fidelity.
</div>
</div>

## Dependencies

Install requirements from the project root directory:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install -r requirements.txt
```
In case errors show up for missing packages, install them manually.

## Structure

Overall code structure is as follows:

| Folder                 | Description                                                                                  |
|------------------------|----------------------------------------------------------------------------------------------|
| `config/`              | hydra  configs                                                                        |
| `data/`                | processed dataset
| `dataset/`             | pytorch datasets and dataloaders                                                           |
| `docs/`                | project webpage files                                                                        |
| `inference/`           | scripts for inferencing trained model                                           |
| `model/`               | pytorch modules for encoder, decoder and the transformer              |
| `pretrained/` | pretrained models on shapenet chairs and tables |
| `runs/`                | model training logs and checkpoints go here in addition to wandb                            |
| `trainer/`             | pytorch-lightning module for training                                                        | 
| `util/`                | misc utilities for positional encoding, visualization, logging etc.                      |

## Pre-trained Models and Data

Download the pretrained models and the data from [here](https://drive.google.com/drive/folders/1Gzuxn6c1pguvRWrsedmCa9xKtest8aC2?usp=drive_link). Place them in the project, such that trained models are in `pretrained/` directory and data is in `data/shapenet` directory. 

### Running inference

To run inference use the following command

```bash
python inference/infer_meshgpt.py <ckpt_path> <sampling_mode> <num_samples>
```

Examples:

```bash
# for chairs
python inference/infer_meshgpt.py pretrained/transformer_ft_03001627/checkpoints/2287-0.ckpt beam 25

# for tables
python inference/infer_meshgpt.py pretrained/transformer_ft_04379243/checkpoints/1607-0.ckpt beam 25
```

## Training

For launching training, use the following command from project root

```
# vocabulary
python trainer/train_vocabulary.py <options> vq_resume=<path_to_vocabulary_ckpt>

# transformer
python trainer/train_transformer.py <options> vq_resume=<path_to_vocabulary_ckpt> ft_category=<category_id> ft_resume=<path_to_base_transformer_ckpt>
```

Some example trainings:

#### Vocabulary training
```bash
python trainer/train_vocabulary.py batch_size=32 shift_augment=True scale_augment=True wandb_main=True experiment=vq128 val_check_percent=1.0 val_check_interval=5 overfit=False max_epoch=2000 only_chairs=False use_smoothed_loss=True graph_conv=sage use_point_feats=False num_workers=24 n_embed=16384 num_tokens=131 embed_levels=2 num_val_samples=16 use_multimodal_loss=True weight_decay=0.1 embed_dim=192 code_decay=0.99 embed_share=True distribute_features=True
```
#### Base transformer training
```bash 

# run over multiple GPUs (recommended GPUs >= 8), if you have a good budget, can use higher gradient_accumulation_steps

python trainer/train_transformer.py wandb_main=True batch_size=8 gradient_accumulation_steps=8 max_val_tokens=5000 max_epoch=2000 sanity_steps=0 val_check_interval=1 val_check_percent=1 block_size=4608 model.n_layer=24 model.n_head=16 model.n_embd=768 model.dropout=0 scale_augment=True shift_augment=True num_workers=24 experiment=bl4608-GPT2_m24-16-768-0_b8x8x8_lr1e-4 use_smoothed_loss=True num_tokens=131 vq_resume=<path_to_vocabulary_ckpt> padding=0
```
#### Transformer finetuning
```bash

# run over multiple GPUs (recommended GPUs >= 8), if you have a good budget, can use higher gradient_accumulation_steps

python trainer/train_transformer.py wandb_main=True batch_size=8 gradient_accumulation_steps=8 max_val_tokens=5000 max_epoch=2400 sanity_steps=0 val_check_interval=8 val_check_percent=1 block_size=4608 model.n_layer=24 model.n_head=16 model.n_embd=768 model.dropout=0 scale_augment=True shift_augment=True num_workers=24 experiment=bl4608-GPT2_m24-16-768-0_b8x8x8_FT04379243 use_smoothed_loss=True num_tokens=131 vq_resume=<path_to_vocabulary_ckpt> padding=0 num_val_samples=4 ft_category=04379243 ft_resume=<path_to_base_transformer_ckpt> warmup_steps=100

```

## License
<a property="dct:title" rel="cc:attributionURL" href="https://nihalsid.github.io/mesh-gpt/">MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://nihalsid.github.io/">Mohd Yawar Nihal Siddiqui</a> is licensed under [Automotive Development Public Non-Commercial License Version 1.0](LICENSE), however portions of the project are available under separate license terms: e.g. NanoGPT code is under MIT license.

## Citation

If you wish to cite us, please use the following BibTeX entry:

```BibTeX
@InProceedings{siddiqui_meshgpt_2024,
    title={MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers},
    author={Siddiqui, Yawar and Alliegro, Antonio and Artemov, Alexey and Tommasi, Tatiana and Sirigatti, Daniele and Rosov, Vladislav and Dai, Angela and Nie{\ss}ner, Matthias},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
}

```