# MultiModalQA
This repo contains the code for running the baselines of the ICLR 2021 paper [MultiModalQA: Complex Question Answering over Text, Tables and Images](https://arxiv.org/abs/2104.06039).

## Setup
Our code is tested on Python 3.7+ and PyTorch 1.5.1.
Python's virtual or Conda environments are recommended for running our code.
Run the following command to install the required libraries.
```bash
pip install -r requirements.txt
```
Our code also relies on [APEX](https://github.com/NVIDIA/apex#linux), [Vilbert](https://github.com/facebookresearch/vilbert-multi-task#repository-setup) and [MaskRCNN](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md) to run.
Please refer to their individual documentation for installation.
Note that if you encounter errors about `AT_CHECK`, probably [this](https://github.com/facebookresearch/maskrcnn-benchmark/issues/1307) can solve the problem.

## Downloading data and models

Run the following script to download all MMQA data and the checkpoints for our baseline models.
```bash
./scripts/download.sh
```

## Running inference

To evaluate our *implicit_decomp* model on the dev set, you can run the following command:
```bash
python pipeline.py \
  --qtype_model_dir downloads/checkpoints/qtype_clf_models/ \
  --text_qa_model_dir downloads/checkpoints/text_qa_models/implicit_decomp/ \
  --table_qa_model_dir  downloads/checkpoints/table_qa_models/implicit_decomp/ \
  --image_qa_model_path downloads/checkpoints/image_qa_models/implicit_decomp/model.pt \
  --image_qa_vocab downloads/checkpoints/image_qa_models/vocab.pickle \
  --image_features_dir downloads/checkpoints/img_features/ \
  --data_dir downloads/MMQA/ \
  --test_file MMQA_dev.jsonl  \
  --output_dir output/results/implicit_decomp_dev/ \
  --do_lower_case \
  --method implicit_decomp
```
By adding the `--predict_only` argument, the same code can be used to predict answers for the test set without evaluating the final metrics.
We also have a script for evaluating all of our baselines (`implicit_decomp`, `auto-routing`, `context-only`) together on the dev set:
```bash
./scripts/run_pipeline.sh
```
The dev performance is as follows:

|                | Single-Modality | Multi-Modality |      All      |
|:--------------:|:---------------:|:--------------:|:-------------:|
|  Context-only  |   7.94 / 10.73  |   6.63 /  9.00 |  7.41 / 10.03 |
|   AutoRouting  |  51.68 / 58.48  |  34.18 / 40.18 | 44.65 / 51.13 |
| ImplicitDecomp |  51.60 / 58.35  |  44.59 / 51.19 | 48.79 / 55.48 |

## Training Models
Our models for different modalities are trained separately.
Please refer to the individual `README.md` in the `text/table/image_qa` folder for training the corresponding model.
