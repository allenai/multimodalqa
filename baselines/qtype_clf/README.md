## Question Type Classifier
This directory includes the code for our question type classifier. 

### Preprocessing
1. Download the MULTIMODALQA data (question file only) from [here](https://github.com/allenai/multimodalqa/tree/master/dataset).

```
mkdir multimodalqa_data
cd multimodalqa_data
wget https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_train.jsonl.gz
wget https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_dev.jsonl.gz
```

2. Unzip the downloaded data.
```
gunzip MMQA_train.jsonl.gz
gunzip MMQA_dev.jsonl.gz
cd ..
```

3. Convert MULTIMODALQA's question file to our modal classification input data. 

```
python3 convert_to_modal_classification_data.py \
    --train_data multimodalqa_data/MMQA_train.jsonl \
    --dev_data multimodalqa_data/MMQA_dev.jsonl \
    --output_data_dir modal_classification_data
```

### Training
To fine-tune a question type classification model, please run the command below after the preprocessing.

```
python run_modal_classifier.py \
--data_dir modal_classification_data \
--model_type bert --model_name_or_path bert-large-uncased \
--do_lower_case --do_eval --do_train --task_name modal 
--output_dir output_modal_classification_bert_large \
--max_seq_length 128 --per_gpu_train_batch_size 12   \
--learning_rate 2e-5 \
--num_train_epochs 3.0 
```

The output result (`accuracy`, `preds` and `output`) will be stored into `OUTPUT_DIR/eval_outputs_results.json` (e.g., `output_modal_classification_bert_large/eval_outputs_results.json` in the example command above). 



