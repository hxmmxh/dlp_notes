- [总览](#总览)
- [基础名词](#基础名词)
- [基础算法](#基础算法)
  - [tokenize](#tokenize)
  - [Embedding](#embedding)
  - [beam search](#beam-search)
- [attention详解](#attention详解)
- [Transform详解](#transform详解)
  - [流程](#流程)
  - [Embedding](#embedding-1)
  - [PE, postion encoding](#pe-postion-encoding)
  - [relative postional embedding](#relative-postional-embedding)

https://kexue.fm/archives/4765

https://www.cnblogs.com/gongyanzh/p/12485587.html

# 总览
https://aistudio.baidu.com/aistudio/course/introduce/2060

https://cloud.tencent.com/developer/article/2026478

# 基础名词
- token
  - tokenization（分词）
  - 分词就是将句子、段落、文章这类型的长文本，分解为以 字词（token） 为单位的数据结构。
  -  比方说，在句子 “我很开心” 中，利用中文分词得到的列表是 {“我”，“很”，“开心”}，列表中的每一个元素代表一个token。
  -  而论文中的token representation表达把文本分词后每个词表示成向量。
- OOV
  - 在自然语言文本处理的时候，我们通常会有一个字词库（vocabulary），它来源于训练数据集。
  - 当然，这个词库是有限的。当以后你有新的数据集时，这个数据集中有一些词并不在你现有的vocabulary里，我们就说这些词汇是out-of-vocabulary，简称OOV

# 基础算法

## tokenize
https://cloud.tencent.com/developer/article/1865689

tokenize的目标是把输入的文本流，切分成一个个子串，每个子串相对有完整的语义，便于学习embedding表达和后续模型的使用

## Embedding
http://mantchs.com/2019/08/22/NLP/Word%20Embeddings/

## beam search
https://blog.csdn.net/xyz1584172808/article/details/89220906
https://zhuanlan.zhihu.com/p/36029811?group_id=972420376412762112

# attention详解
https://blog.51cto.com/u_11466419/5509561
https://zhuanlan.zhihu.com/p/420820453



# Transform详解

## 流程

- Inputs X[batch_size, sequence_length], batch_size指的是句子数, sequence_length是输入的句子中最长的句子的字数
- 通过查表Embedding, 得到`X_embedding[batch_size, sequence_length, embedding_dimension]`,embedding_dimension是词向量的长度


## Embedding
https://blog.csdn.net/weixin_44012382/article/details/113059423

## PE, postion encoding
https://blog.csdn.net/weixin_44012382/article/details/113059423
- pos指的是当前字符在句子中的位置
- dmodel指的是word embedding的长度

- 


## relative postional embedding
https://zhuanlan.zhihu.com/p/344604604?utm_id=0
- NLP的输入一般为（batch, sentence_length, embedding_dim）
  - sentence_length 这句话中有多少个词
  - embedding_dim：权重矩阵的列数，通常为512
    - https://www.jianshu.com/p/e6b5b463cf7b
    - 参考torch.nn.Embedding



https://www.jianshu.com/p/e6b5b463cf7b
https://www.jianshu.com/p/81901d3d3f8e
https://www.jianshu.com/p/2ee8a87bcd29