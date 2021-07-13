export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p output/
mkdir -p logs/

CUDA_VISIBLE_DEVICES=0 python pipeline.py \
  --qtype_model_dir downloads/checkpoints/modal_clf_models/output_modal_classification_20210310_final/ \
  --text_qa_model_dir downloads/checkpoints/text_qa_models/roberta_large_context_only_squad_ft_on_mm2/ \
  --table_qa_model_dir downloads/checkpoints/table_qa_models/roberta_large/context_only/checkpoint-15000/ \
  --image_qa_model_path "downloads/checkpoints/image_qa_models/vilbert_context_only_10|03|2021_22:15:54_final.pt" \
  --image_qa_vocab downloads/checkpoints/image_qa_models/vocab.pickle \
  --image_feature_dump downloads/checkpoints/image_qa_models/dev_test_img_info.pickle \
  --input_file downloads/MMQA/MMQA_dev.jsonl  \
  --output_dir output/results/roberta_context_only_final_dev/ \
  --do_lower_case \
  --method "auto_routing" \
  --context_only \
  &> logs/testing_roberta_context_only_final_dev.log &


CUDA_VISIBLE_DEVICES=2 python pipeline.py \
  --qtype_model_dir downloads/checkpoints/modal_clf_models/output_modal_classification_20210310_final/ \
  --text_qa_model_dir downloads/checkpoints/text_qa_models/roberta_large_squad_auto_mm2_ft_final_20200311/ \
  --table_qa_model_dir downloads/checkpoints/table_qa_models/roberta_large/plain_format/checkpoint-17500/ \
  --image_qa_model_path "downloads/checkpoints/image_qa_models/vilbert_auto_routing_10|03|2021_18:08:53_final.pt" \
  --image_qa_vocab downloads/checkpoints/image_qa_models/vocab.pickle \
  --image_feature_dump downloads/checkpoints/image_qa_models/dev_test_img_info.pickle \
  --input_file downloads/MMQA/MMQA_dev.jsonl  \
  --output_dir output/results/roberta_auto_routing_final_dev/ \
  --do_lower_case \
  --method "auto_routing" \
  &> logs/testing_roberta_auto_routing_final_dev.log &


CUDA_VISIBLE_DEVICES=4 python pipeline.py \
  --qtype_model_dir downloads/checkpoints/modal_clf_models/output_modal_classification_20210310_final/ \
  --text_qa_model_dir downloads/checkpoints/text_qa_models/mm2_roberta_large_implicit_decomp_final_20200311/ \
  --table_qa_model_dir  downloads/checkpoints/table_qa_models/roberta_large/implicit_decomp/checkpoint-20000/ \
  --image_qa_model_path "downloads/checkpoints/image_qa_models/vilbert_implicit_decomp_10|03|2021_13:39:08_final.pt" \
  --image_qa_vocab downloads/checkpoints/image_qa_models/vocab.pickle \
  --image_feature_dump downloads/checkpoints/image_qa_models/dev_test_img_info.pickle \
  --input_file downloads/MMQA/MMQA_dev.jsonl  \
  --output_dir ../../multimodalqa/baselines/output/results/roberta_implicit_decomp_final_dev/ \
  --do_lower_case \
  --method "implicit_decomp" \
  &> logs/testing_roberta_implicit_decomp_final_dev.log &


