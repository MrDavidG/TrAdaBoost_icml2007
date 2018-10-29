# TrAdaBoost_icml2007

此代码是实现了icml 2007论文Boosting for transfer learning中的TrAdaBoost方法，并且附带实现了其实验部分纯数值型特征的mushroom数据集。

其中数据集已经经过粗略的编码，用数值来代替特征，详细数据处理的在dataset_mushroom.py里。

## 实验设置

试验参数按照论文icml07中实验的设置

* 迭代N=100轮
* 采用的分类器为sklearn库的linear svm模型，详细参数见icml07.py

## 仍存在的问题：

在mushroom上的实验并没有收敛，最后的加权error rate仍然在0.49+左右，并且最后模型的预测结果并不是很好。

欢迎探讨或是纠正代码中存在的错误。
