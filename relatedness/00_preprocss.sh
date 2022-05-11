python ./src/process_data.py \
    --input ../../data/relate/LinkPrediction_Dataset/train.csv  \
    --output ../../data/relate/LinkPrediction_Dataset/processed_train.csv 2>&1| tee ./precess_train.log
python ./src/process_data.py \
    --input ../../data/relate/LinkPrediction_Dataset/dev.csv  \
    --output ../../data/relate/LinkPrediction_Dataset/processed_dev.csv 2>&1| tee ./precess_dev.log
python ./src/process_data.py \
    --input ../../data/relate/LinkPrediction_Dataset/test.csv  \
    --output ../../data/relate/LinkPrediction_Dataset/processed_test.csv 2>&1| tee ./precess_test.log
