# BetaE-Simple-Reproduction-in-Konwledge-Graph-Reasoning
BetaE论文（2020）：Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs

Beta Embedding：BETAE的主要思路是将所有的实体（entitles）和查询（query）都嵌入成一个在[0, 1]区间内的Beta分布，通过有界的概率分布来对entitles和query进行建模。是首个能够处理一整套一阶逻辑运算的方法，包括合取（∧）、析取（∨）和否定（¬）。

本人对这篇论文进行了详细阅读并做出复现，提供大家学习并找出问题。

阅读笔记在https://blog.csdn.net/Mitchell_Yee/article/details/156570677?spm=1001.2014.3001.5501
