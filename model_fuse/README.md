## 模型融合策略
1. 参与最终融合结果的模型一共有7个，分别是：1个官方基线模型Gector、2个指针生成网络（Pointer-Generator-Network）、1个基于Bert+CRF的序列标注模型、1个基于Bert+BiLstm+CRF的序列标注模型以及2个拼写纠错模型
2. 融合的思路主要是：

    1）多个模型融合结果中存在同一个位置有多个错误类型的情况，这种情况可以采用参与判定模型的数目以及计算修改前后的困惑度降低值来进行筛选；
    
    2）多个模型融合结果中存在位置有重叠的情况，这时可以采用参与判定模型的数目来筛选；
3. 详细的融合策略见`model_fuse.ipynb`

## 参赛系统用于最终结果的融合结果
### 语病检测模型
    Bert_CRF.txt 
    Bert_Bilstm_CRF.txt
### 语病纠错模型
    Gector.txt
    Pointer_Generator_Net.txt
    PGN_二阶段.txt
### 错别字模型
    病句S.txt
    易错字S.txt