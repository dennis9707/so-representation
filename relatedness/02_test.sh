CUDA_VISIBLE_DEVICES=0,4,5,6,7 python ./src/test.py \
--model_name_or_path ./result/\
--tokenizer_name microsoft/codebert-base \
--per_device_eval_batch_size 40 \
--test_file ../../data/relate/LinkPrediction_Dataset/test_data.csv \
--output_dir ./predictions/A\
--max_seq_length 512 \
--log_level info \
--do_predict \
"$@" 2>&1| tee ./test_relate.log
