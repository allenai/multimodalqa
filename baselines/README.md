# MultiModalQA
This repo contains the code for running the baselines of the ICLR 2021 paper [https://arxiv.org/abs/2104.06039](MultiModalQA: Complex Question Answering over Text, Tables and Images).

## Setup
Our code is tested on Python 3.7+ and PyTorch 1.5.1.
Python's virtual or Conda environments are recommended for running our code.
Run the following command to install the required libraries.
```bash
pip install -r requirements.txt
```
Our code also relies on [https://github.com/NVIDIA/apex](APEX), [https://github.com/facebookresearch/vilbert-multi-task](Vilbert) and [https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md](MaskRCNN) to run.
Please refer to their individual documentation for installation.
Note that if you encounter errors about `AT_CHECK`, probably [https://github.com/facebookresearch/maskrcnn-benchmark/issues/1307](this) can solve the problem.

## Downloading data and models

Run the following script to download all MMQA data and the checkpoints for out baseline models.
```bash
    ./scripts/download.sh
```

## Running inference
First, run the following command to extract and cache the features for all images.
```bash
```
Then, with the following command, you can get the dev results for the `context-only`, `auto-routing` and `implicit_decomp` models described in our paper:
```bash
    ./scripts/run_pipeline.sh
```
By adding the `--predict_only` argument, the same code can be used to predict answers for the test set without evaluating the final metrics.

## Training Models
Our models for different modalities are trained separately.
Please refer to individual `README.md` in the `text/table/image_qa` folder for training them.
