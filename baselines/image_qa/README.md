
## Image - related code

### Install dependencies

```
conda env create -f environment.yml
source activate multimodalqa_image
```

Follow the installation instructions for ViLBERT at `deps/vilbert-multi-task`

### Run training

```
 python baselines/image_qa/main.py
```

A full list of arguments is present in the file main.py. This file will construct and cache the dataset and image features automatically if it is not already done. Resulting models will be stored at `baselines/image_qa/checkpoints`.

### Inference

For an example of how to run inference with a trained model, see `baselines/image_qa/inference.py`.