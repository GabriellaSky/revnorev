python run_svm.py --input_data './data/sample.csv' \
--C '0.1' \
--model 'google/electra-base-discriminator' \
--context 'parent_1_text' \
--output_dir './output/svm/base/' \
--exp_setup 'random'


python run_multiclass_svm.py --input_data './data/sample.csv' \
--C '0.1' \
--model 'bert-base-cased' \
--context 'parent_1_text' \
--output_dir './output/svm/base/'


python run_hf.py --input_data './data/sample.csv' \
--n_epochs 5 \
--pretrained_model 'bert-base-cased' \
--context 'title' \
--output_dir './output/svm/base/' \
--batch_size 4 \
--warmup_steps 20 \
--eval_steps 20 \
--max_seq_len 256 \
--exp_setup 'cc'

python run_multiclass_hf.py --input_data './data/sample.csv' \
--n_epochs 5 \
--pretrained_model 'bert-base-cased' \
--context 'title' \
--output_dir './output/svm/base/' \
--batch_size 4 \
--warmup_steps 20 \
--eval_steps 20 \
--max_seq_len 256

python run_depth_hf.py  --input_data '../final_merged.csv' \
--n_epochs 1 \
--pretrained_model 'bert-base-cased' \
--context 'title' \
--output_dir './output/svm/base/' \
--batch_size 16 \
--warmup_steps 1000 \
--eval_steps 1000 \
--max_seq_len 128
