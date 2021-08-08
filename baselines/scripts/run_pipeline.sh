export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p output/
mkdir -p logs/

# for the implicit decomp baseline
CUDA_VISIBLE_DEVICES=0 python pipeline.py \
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
  --method implicit_decomp \
  &> logs/evaluate_implicit_decomp_dev.log &

# for the auto routing baseline
CUDA_VISIBLE_DEVICES=1 python pipeline.py \
  --qtype_model_dir downloads/checkpoints/qtype_clf_models/ \
  --text_qa_model_dir downloads/checkpoints/text_qa_models/auto_routing/ \
  --table_qa_model_dir downloads/checkpoints/table_qa_models/auto_routing/ \
  --image_qa_model_path downloads/checkpoints/image_qa_models/auto_routing/model.pt \
  --image_qa_vocab downloads/checkpoints/image_qa_models/vocab.pickle \
  --image_features_dir downloads/checkpoints/img_features/ \
  --data_dir downloads/MMQA/ \
  --test_file MMQA_dev.jsonl  \
  --output_dir output/results/auto_routing_dev/ \
  --do_lower_case \
  --method auto_routing \
  &> logs/evaluate_auto_routing_dev.log &

# for the context only baseline
CUDA_VISIBLE_DEVICES=2 python pipeline.py \
  --qtype_model_dir downloads/checkpoints/qtype_clf_models/ \
  --text_qa_model_dir downloads/checkpoints/text_qa_models/context_only/ \
  --table_qa_model_dir downloads/checkpoints/table_qa_models/context_only/ \
  --image_qa_model_path downloads/checkpoints/image_qa_models/context_only/model.pt \
  --image_qa_vocab downloads/checkpoints/image_qa_models/vocab.pickle \
  --image_features_dir downloads/checkpoints/img_features/ \
  --data_dir downloads/MMQA/ \
  --test_file MMQA_dev.jsonl  \
  --output_dir output/results/context_only_dev/ \
  --do_lower_case \
  --method auto_routing \
  --context_only \
  &> logs/evaluate_context_only_dev.log &
