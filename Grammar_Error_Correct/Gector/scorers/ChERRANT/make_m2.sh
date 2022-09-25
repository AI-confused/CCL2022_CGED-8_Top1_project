INPUT_FILE=/home/wangzhihao/jingsai/cltc/MuCGEC/data/test_data/2021_test_scr.txt
OUTPUT_FILE=/home/wangzhihao/jingsai/cltc/MuCGEC/models/seq2edit-based-CGEC/exps/seq2edit_lang8/results/MuCGEC_test.output
HYP_PARA_FILE=./samples/MuCGEC_test.hyp.para

paste $INPUT_FILE $OUTPUT_FILE | awk '{print NR"\t"$p}' > $HYP_PARA_FILE

HYP_M2_FILE=./samples/MuCGEC_test.hyp.m2.char

python parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char