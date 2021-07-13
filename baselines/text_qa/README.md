## Text QA model 
This directory contains code for our text QA model that takes a paragraph and jointly predicts (1) span and (2) answer type classification (`span`, `no answer`, `yes`, `no`). At inference, our model outputs the span answer from the paragraph with the lowest `no_answer_probability`. Please see the details of the model in our paper. 


### Preprocessing
1. Download the text data as well as our question data.
```
mkdir multimodalqa_data
cd multimodalqa_data
wget https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_train.jsonl.gz
wget https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_dev.jsonl.gz
wget https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_texts.jsonl.gz
```

2. Unzip the downloaded files.
```
unzip MMQA_train.jsonl.gz
unzip MMQA_dev.jsonl.gz
unzip MMQA_texts.jsonl.gz
cd ..
```

3. Convert MULTIMODALQA data into our text QA format.
The following command converts our MultimodalQA data into our text QA format, which is consistent with [the SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). 

```
python3 convert_data_into_squad_format.py  \
    --train_data_fp multimodalqa_data/MMQA_train.jsonl \
    --dev_data_fp multimodalqa_data/MMQA_dev.jsonl \
    --doc_data multimodalqa_data/MMQA_texts.jsonl \
    --output_data_dir MMQA_squad_format \
    --mode implicit_decomp
```

You can change the `mode` option to generate the preprocessed files for different baselines.
- `auto` for the auto routing baseline
- `context_only` for the question only baseline
- `implicit_decomp` for our implicit decomposition model

### Training
We first fine-tune our model on SQuAD v.2 data, and then fine-tune the model on our MMQA. 

1. Download the SQuAD 2 data files.  

```
mkdir squad2_data
cd squad2_data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

2. Train our Text QA model by running the command below. We use `roberta-large` as our base pre-trained model.
```
python run_squad.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --logging_steps 5000 \
    --version_2_with_negative  \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length 384  \
    --doc_stride 128 \
    --output_dir roberta_large_squad \
    --per_gpu_train_batch_size=8 \
    --save_steps 5000 \
    --train_file /path/to/squad2/train_data \
    --predict_file /path/to/squad2/dev_data \
    --do_train --do_eval
```

3. Run the command below to train the text QA model on our MMQA dataset. 

```
python run_squad.py \
    --model_type roberta \
    --model_name_or_path roberta_large_squad \
    --logging_steps 5000 \
    --version_2_with_negative \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length 384  \
    --doc_stride 128  \
    --output_dir roberta_large_multimodalqa \
    --per_gpu_train_batch_size=8 \
    --save_steps 5000 \
    --train_file MMQA_squad_format/MultiModalQA_train_converted_to_squad.mode=implicit_decomp_model=roberta.json \
    --predict_file MMQA_squad_format/MultiModalQA_dev_converted_to_squad.mode=implicit_decomp_model=roberta.json \
    --do_eval --do_train
```
