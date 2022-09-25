# Step1. Data Preprocessing

## Download Structbert
if [ ! -f ./plm/chinese-struct-bert-large/pytorch_model.bin ]; then
    wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/ch_model
    mv ch_model ./plm/chinese-struct-bert-large/pytorch_model.bin
fi


CUDA_DEVICE=6
MODEL_DIR=./exps/seq2edit_lang8
PRETRAIN_WEIGHTS_DIR=./plm/chinese-struct-bert-large
VOCAB_PATH=./data/output_vocabulary_chinese_char_hsk+lang8_5

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
OUTPUT_FILE=$RESULT_DIR"/cged2022-test.output"

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
