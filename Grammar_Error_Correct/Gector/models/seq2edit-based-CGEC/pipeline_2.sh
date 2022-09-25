# Step1. Data Preprocessing

## Download Structbert
if [ ! -f ./plm/chinese-struct-bert-large/pytorch_model.bin ]; then
    wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/ch_model
    mv ch_model ./plm/chinese-struct-bert-large/pytorch_model.bin
fi

# Tokenize
SRC_FILE=../../data_track2/alldata/data_train_14_to_18_src_without_kongge.txt  # 每行一个病句
TGT_FILE=../../data_track2/alldata/data_train_14_to_18_trg_without_kongge.txt   # 每行一个正确句子，和病句一一对应
if [ ! -f $SRC_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $SRC_FILE > $SRC_FILE".char"  # 分字
fi
if [ ! -f $TGT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $TGT_FILE > $TGT_FILE".char"  # 分字
fi

# Generate label file
LABEL_FILE=../../data_track2/alldata/data_train_14_to_18.label  # 训练数据
if [ ! -f $LABEL_FILE ]; then
    python ./utils/preprocess_data.py -s $SRC_FILE".char" -t $TGT_FILE".char" -o $LABEL_FILE --worker_num 32
    shuf $LABEL_FILE > $LABEL_FILE".shuf"
fi

# Step2. Training
CUDA_DEVICE=2
SEED=1

DEV_SET=../../data_track2/alldata/dev.label
MODEL_DIR=./exps/seq2edit_lang8
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi

PRETRAIN_WEIGHTS_DIR=./plm/chinese-struct-bert-large

cp ./pipeline_2.sh $MODEL_DIR/src_bak

VOCAB_PATH=./data/output_vocabulary_chinese_char_hsk+lang8_5


## Unfreeze encoder,这一步是经过第一步的lang8数据训练后，在使用往年数据进行训练
LR=1e-5
BATCH_SIZE=48
ACCUMULATION_SIZE=4
MODEL_NAME=Best_Model_Stage_3
EPOCH=20
PATIENCE=3


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --tune_bert 1\
                --train_set $LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $BATCH_SIZE\
                --n_epoch $EPOCH\
                --lr $LR\
                --accumulation_size $ACCUMULATION_SIZE\
                --patience $PATIENCE\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --pretrain_folder $MODEL_DIR\
                --pretrain "Best_Model_Stage_2"\
                --seed $SEED


# Step3. Inference
MODEL_PATH=$MODEL_DIR"/Best_Model_Stage_3.th"
RESULT_DIR=$MODEL_DIR"/results"

INPUT_FILE=../../data_track2/2022test/cged2022-test.txt # 输入文件
if [ ! -f $INPUT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $INPUT_FILE > $INPUT_FILE".char"  # 分字
fi
if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi
OUTPUT_FILE=$RESULT_DIR"/cged2022-test_stage_3.output"

echo "Generating..."
SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --input_file $INPUT_FILE".char"\
                  --output_file $OUTPUT_FILE --log

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
