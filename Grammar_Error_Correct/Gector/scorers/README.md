M2Scorer下载链接：https://github.com/nusnlp/m2scorer

Download link of M2Scorer: https://github.com/nusnlp/m2scorer


王智浩：
这一步，首先将预测结果转换成m2文件，使用的是make_m2.sh，输入是源文件、预测文件，格式为不带任何空格的句子。这一步重复6遍，计算3个seq2edit和3个seq2seq的结果。

然后，使用ensemble.sh 对第一步的输出进行处理，得到最终的融合结果。


如果我们想用这个计算一个能提交赛道四的答案，那么直接使用demo_track4.sh