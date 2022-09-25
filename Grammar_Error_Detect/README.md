# 语法检测模型
## Bert+CRF
### 模型方案
1. 基于上一届冠军方案的思路，我们沿用了Bert+CRF的模型，采用Macbert-large+CRF，序列标注标签为BIEOS方案，一共17个类别

`'O', 'S-B', 'M-B', 'W-B', 'R-B', 'S-I', 'M-I', 'W-I', 'R-I', 'S-E', 'M-E', 'W-E', 'R-E', 'S-S', 'M-S', 'W-S', 'R-S'`

2. 考虑到序列标注问题的特殊性，我们对Bert分词器的tokenize功能进行了继承重写，直接用list(text)替换，确保不会因为分词导致解码的问题

3. 该任务代码基于深度学习训练框架[easy-task](https://github.com/AI-confused/easy_task)编写，需要在环境中pip install easy-task==0.0.30，以便与运行任务代码库

4. 整体训练策略为：先用lang8数据预训练，再基于前者最好的模型，用历年数据训练集微调，2个阶段的训练超参数如下：

| 训练阶段| 数据集  | lr     |max_seq_len|batch_size|dropout|scheduler|warmup|epoch|GPU卡数|梯度累积|seed|
|:-------:|:-------:|:-----:|:-------:|:--------:|:------:|:-----:|:--------:|:-----:|:--------:|:--------:|:--------:|
|预训练| lang8   |  1e-5  |256        |256       |0.1    |   cosine|0.0|10   |8|1|99|
|微调| 历年训练集|     5e-6   |256|256|0.3|cosine|0.0|10|8|1|99|

5. 模型训练过程中，每半个epoch验证模型一次，根据最高的指标保存最佳模型，同样会保存每个验证阶段的模型（配置文件中save_cpt_flag设置为2）

6. 保存最佳模型的参考指标为**detect_f1 + iden_f1 + posidentification_f1 - FPR**，这样保存的模型不会偏向于某一个指标，而是趋向于总分最高，因为在初始的实验中发现：如果按照序列标注的posidentification_f1指标来保存模型的话，最佳模型的posi偏高，但是FPR也会偏高，这样总分其实是会下降的

### 如何运行
#### Train
1. 修改`config/bert_crf_train.yml`配置文件中的超参数(**超参数含义见配置文件注释**)
2. `run_task_bert_crf.py`入口文件中第10和11行选择该配置文件的那一行
3. 进入`Bert+CRF`工作目录，运行入口文件`python3 src/run_task_bert_crf.py`
4. 运行任务后，在工作目录下的`output/"task_name"/Log/`会生成同步的训练日志文件，包含训练过程中的每个验证结果的指标、保存的badcase文件地址、保存的最佳模型(模型文件名包含dev.0)和每个验证checkpoint模型(模型文件名包含checkpoint)地址
#### Test
1. 修改`config/bert_crf_predict.yml`配置文件中的超参数
2. `run_task_bert_crf.py`入口文件中第10和11行选择该配置文件的那一行，第27行中resume_model_path赋值为第一步lang8预训练任务的最佳模型地址
3. `python3 src/run_task_bert_crf.py`
4. 运行任务后，在工作目录下的`output/"task_name"/Log/`会生成同步的预测日志文件，包含测试集的预测结果(excel格式)文件地址
5. 将测试集预测结果通过`convert_predict.ipynb`转换为提交格式，由于该模型为检测模型，没有S和M类型的纠正结果，因此这里只有检测相关的指标体现
### 实验记录
#### lang8预训练阶段(历年20+21测试集作为验证集)
|数据集|  FPR |   detect_f1 |   identification_f1 |   position_f1 |
|:---:|:---:|:---:|:---:|:---:|
|20+21测试集|0.2144|0.7989|0.5301|0.3004|
|22测试集|0.2212|0.7836|0.5135|0.3058|
#### 历年数据微调阶段(历年20+21测试集作为验证集)
|数据集|  FPR |   detect_f1 |   identification_f1 |   position_f1 |
|:---:|:---:|:---:|:---:|:---:|
|20+21测试集|0.1642|0.7977|0.5568|0.3469|
|22测试集|0.1740|0.7723|0.5093|0.3339|
## Bert+BiLSTM+CRF
### 模型方案
1. 基于Bert+CRF的思路，我们在中间添加了一个BiLSTM层，尝试下能否给CRF提供更好的发射矩阵，模型采用Macbert-large+BiLSTM+CRF，序列标注标签为BIEOS方案，但是这里尝试了新的方案：一共13个类别，把一些不可能出现的标签去掉了（M类型只有Single，W类型没有Single）

`'O', 'S-B', 'W-B', 'R-B', 'S-I', 'W-I', 'R-I', 'S-E', 'W-E', 'R-E', 'S-S', 'M-S', 'R-S'`

2. 考虑到序列标注问题的特殊性，同样对Bert分词器的tokenize功能进行了继承重写，直接用list(text)替换，确保不会因为分词导致解码的问题

3. 该任务代码基于深度学习训练框架[easy-task](https://github.com/AI-confused/easy_task)编写，需要在环境中pip install easy-task==0.0.30，以便与运行任务代码库

4. 整体训练策略为：先用lang8数据预训练，再基于前者最好的模型，用历年数据训练集微调，2个阶段的训练超参数如下：

| 训练阶段| 数据集  | lr     |max_seq_len|batch_size|dropout|scheduler|warmup|epoch|GPU卡数|梯度累积|seed|
|:-------:|:-------:|:-----:|:-----:|:-------:|:--------:|:------:|:-----:|:--------:|:-----:|:--------:|:--------:|
|预训练| lang8   |  1e-5  |256        |256       |0.1    |   cosine|0.0|10   |4|2|99|
|微调| 历年训练集|     5e-6   |256|256|0.3|linear|0.1|10|2|4|99|

5. 模型训练过程中，每半个epoch验证模型一次，根据最高的指标保存最佳模型，同样会保存每个验证阶段的模型（配置文件中save_cpt_flag设置为2）

6. 保存最佳模型的参考指标为**detect_f1 + iden_f1 + posidentification_f1 - FPR**，这样保存的模型不会偏向于某一个指标，而是趋向于总分最高，因为在初始的实验中发现：如果按照序列标注的posidentification_f1指标来保存模型的话，最佳模型的posi偏高，但是FPR也会偏高，这样总分其实是会下降的
### 如何运行
1. 修改config/bert_bilstm_crf_train.yml配置文件中的超参数
2. run_task_bert_bilstm_crf.py入口文件中第10和11行选择该配置文件的那一行
3. 进入`Bert+BiLSTM+CRF`工作目录，运行入口文件`python3 src/run_task_bert_bilstm_crf.py`
4. 运行任务后，在工作目录下的output/"task_name"/Log/会生成同步的训练日志文件，包含训练过程中的每个验证结果的指标、保存的badcase文件地址、保存的最佳模型(模型文件名包含dev.0)和每个验证checkpoint模型(模型文件名包含checkpoint)地址
#### Test
1. 修改config/bert_crf_bilstm_predict.yml配置文件中的超参数
2. run_task_bert_crf.py入口文件中第10和11行选择该配置文件的那一行，第27行中resume_model_path赋值为第一步lang8预训练任务的最佳模型地址
3. `python3 src/run_task_bert_bilstm_crf.py`
4. 运行任务后，在工作目录下的output/"task_name"/Log/会生成同步的预测日志文件，包含测试集的预测结果(excel格式)文件地址
5. 将测试集预测结果通过`convert_predict.ipynb`转换为提交格式，由于该模型为检测模型，没有S和M类型的纠正结果，因此这里只有检测相关的指标体现
### 实验记录
#### lang8预训练阶段(历年20+21测试集作为验证集)
|数据集|  FPR |   detect_f1 |   identification_f1 |   position_f1 |
|:---:|:---:|:---:|:---:|:---:|
|20+21测试集|0.1858|0.8007|0.5318|0.3014|
|22测试集|0.2183|0.7780|0.4978|0.3013|
#### 历年数据微调阶段(历年20+21测试集作为验证集)
|数据集|  FPR |   detect_f1 |   identification_f1 |   position_f1 |
|:---:|:---:|:---:|:---:|:---:|
|20+21测试集|0.1799|0.8083|0.5635|0.3514|
|22测试集|0.2094|0.7753|0.5089|0.3298|

## yml配置文件参数信息
    skip_train：训练模型为0，预测模型为1
    exp_dir：输出保存目录
    task_name：任务名称，整个训练任务的输出会保存在`exp_dir/task_name/`
    eval_portion：预测验证集指标的频率，0.5意思是每半个epoch运行一次验证集指标
    bad_case：是否输出bad case结果
    save_cpt_flag：模型保存机制，0是仅保存最佳checkpoint的模型; 1是保存最佳checkpoint和最后一个epoch的模型; 2是保存最佳checkpoint和每个epoch的模型
