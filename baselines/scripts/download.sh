mkdir -p downloads/MMQA/
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_train.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_dev.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_test.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_texts.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_tables.jsonl.gz
wget -P downloads/MMQA https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip
gunzip downloads/MMQA/*.gz
unzip downloads/MMQA/*.zip -d downloads/MMQA/

mkdir -p downloads/vilbert/
wget -P downloads/vilbert https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget -P downloads/vilbert https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
wget -P downloads/vilbert https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin
wget -P downloads/vilbert https://raw.githubusercontent.com/facebookresearch/vilbert-multi-task/master/config/bert_base_6layer_6conect.json
wget -P downloads/vilbert https://raw.githubusercontent.com/facebookresearch/vilbert-multi-task/master/vilbert_tasks.yml

mkdir -p downloads/checkpoints/
wget -P downloads/checkpoints/ https://multimodalqa.s3.us-west-2.amazonaws.com/baseline_checkpoints/qtype_clf_models.tar.gz
wget -P downloads/checkpoints/ https://multimodalqa.s3.us-west-2.amazonaws.com/baseline_checkpoints/text_qa_models.tar.gz
wget -P downloads/checkpoints/ https://multimodalqa.s3.us-west-2.amazonaws.com/baseline_checkpoints/table_qa_models.tar.gz
wget -P downloads/checkpoints/ https://multimodalqa.s3.us-west-2.amazonaws.com/baseline_checkpoints/image_qa_models.tar.gz
for f in downloads/checkpoints/*_models.tar.gz; do tar xvzf "$f" -C downloads/checkpoints; done

wget -P downloads/checkpoints/ https://multimodalqa.s3.us-west-2.amazonaws.com/baseline_checkpoints/img_features.tar.gz
tar xvzf downloads/checkpoints/img_features.tar.gz -C downloads/checkpoints/
