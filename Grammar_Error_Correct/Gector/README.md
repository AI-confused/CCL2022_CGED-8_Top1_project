## 关于数据：
1、其中的lang8数据是来自官方提供的数据集，经过数据处理说明部分提到的去空格、去英文、去繁体等操作
后得到的数据，内容存在lang8_src.txt和lang8_trg.txt；
2、其中的历年数据是14-18年的所有训练集，去空格后存在data_train_14_to_18_src_without_kongge和
data_train_14_to_18_trg_without_kongge中。
3、验证集来自往年数据的14-20年测试集，存在data_test_14_to_20_src.txt和data_test_14_to_20_trg.txt

## 关于训练环境：
32G Tesla V100, CUDA Version: 11.4
运行需要的软件，在requiments.txt

## 训练步骤：
1、运行pipline.sh,生成Best_Model_Stage_1.th和Best_Model_Stage_2.th
2、运行pipeline_2.sh,生成Best_Model_Stage_3.th

## 预测测试集
运行inference.sh

## 转化为官方提交格式
# 官方赛道2提供了baseline，其中CCL2022-CLTC-main/baselines/track2/metric/run_eval.sh提供的数据转化方法，
SRC_PATH= track2/dataset/2022test/cged2022-test.txt （测试集）
HYP_PATH= results/cged2022_test.output（预测的结果）
OUTPUT_PATH = cged2022_test_result.txt

运行 python pair2edits_char.py $SRC_PATH $HYP_PATH > $OUTPUT_PATH

## 备注：
本模型基于GECToR，预训练模型使用chinese-struct-bert-large。
