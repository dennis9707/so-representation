CUDA_VISIBLE_DEVICES=0,4,5,6,7 python ./src/test.py \
--model_name_or_path ./result/April-21-longformer-code-cl-setting2/checkpoint-2000 \
--tokenizer_name dennishe97/longformer-code-cl \
--per_device_eval_batch_size 40 \
--test_file ../../data/relate/LinkPrediction_Dataset/test_data.csv \
--output_dir ./predictions/April-21-longformer-code-cl-setting2/checkpoint-2000 \
--max_seq_length 512 \
--log_level info \
--do_predict \
"$@" 2>&1| tee ./test_relate_mlm-cl-setting2.log
# done 2>&1| tee ./test_relate_mlm-v3-setting2.log