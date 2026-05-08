## 实验准备

去网站[https://github.com/stanfordnlp/GloVe下载Common](https://github.com/stanfordnlp/GloVe下载Common) Crawl 840B预训练词向量权重，转化文本为word2vec格式供gensim加载，run imdb_process.py文件将数据集进行清洗，分词，训练集验证集的划分，并生成文件imdb_glove.pickle3方便以后模型训练进行加载（由于文件过大故没有上传GitHub）。

## 实现训练过程

```
对于所有训练脚本的训练参数均保持原样
```

1. 实验数据的读取，输入数据在本机电脑的相对路径
2. 统一结果输出路径到result文件夹中
3. 由于版本API变更，实验过程中将旧的 datasets.load_metric 修改为 evaluate.load("accuracy")，TrainingArguments evaluation_strategy参数改为 eval_strategy="epoch"。

## 实验中遇到的问题

1. 在imdb_transformer.py的实验中由于review_to_wordlist返回的是字符串不是词列表，导致序列过长导致编码维度冲突，后续在大模型辅助下引入最大序列长度max_length在函数collate_fn截断解决，但是训练之后的结果在kaggle平台测试结果不理想。
2. 训练过程中由于imdb_bert_trainer.py实验时长达到9-10小时完成一轮，本地电脑算力无法支持故放弃训练。

## Kaggle 提交成绩汇总


| 排名  | 提交文件                        | 分数      |
| --- | --------------------------- | ------- |
| 1   | roberta_trainer.csv         | 0.95200 |
| 2   | bert_scratch.csv            | 0.93448 |
| 3   | distilbert_trainer.csv      | 0.92940 |
| 4   | distilbert_native.csv       | 0.91780 |
| 5   | bert_native.csv             | 0.90672 |
| 6   | gru.csv                     | 0.88132 |
| 7   | capsule_lstm.csv            | 0.86732 |
| 8   | Bag_of_Words_model.csv      | 0.84504 |
| 9   | Word2Vec_AverageVectors.csv | 0.83088 |
| 10  | cnnlstm.csv                 | 0.81184 |
| 11  | lstm.csv                    | 0.75692 |
| 12  | cnn.csv                     | 0.54520 |
| 13  | transformer.csv             | 0.50000 |
| 14  | bert_trainer.csv             | 0.93888 |

## 补交实验为bert_trainer，已加在成绩汇总第14行
