CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node 8 ./src/train.py \
    --model_name_or_path microsoft/codebert-base  \
    --train_file ../../data/relate/LinkPrediction_Dataset/processed_train.csv \
    --validation_file ../../data/relate/LinkPrediction_Dataset/processed_dev.csv \
    --test_file ../../data/relate/LinkPrediction_Dataset/processed_test.csv \
    --output_dir ./result/ \
    --num_train_epochs 12 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --log_level info \
    --do_train \
    --do_eval \
    --do_predict \
    --save_steps 500 \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --fp16 \
    --fp16_opt_level O2 \
    "$@" 2>&1| tee ./train.log
