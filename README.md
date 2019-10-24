# Stock_regression
## 深度学习在股价中的研究 
---
前一阵研究股价预测，有些许心得，在这里跟大家分享一下，我们的研究主要分为三部分
1. 传统的统计学习回归（SVR回归）<br>
支持向量机回归应该算得上是比较传统的回归模型了，当然，对于股价这种类时间序列问题，也可以采用ARIMA回归等时间序列的处理方法，在这个阶段我会一一实现这些方法
2. 主流的机器学习回归（LSTM）
3. 深度强化学习（实时调参）
---
## 实验研究
1. 传统统计学习回归模型以及时间序列模型<br>
先简单介绍下普通的常见回归问题：<br>

其中输入的数据通过excel中的wind函数进行提取，选取了如下一些基本面指标、wind的所有技术指标、宏观指标（由于宏观指标的滞后性，有些滞后1期，有些滞后2期）：<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/org_data.jpg)<br>
通过将原始的73维数据使用PCA降维到23维，然后划分最后12个月为验证集，其余为训练集，设置C为100，gamma为0.0001利用SVR进行训练得到对比曲线图如下：<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/SVR_regression.jpg)<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/validation_img.jpg)<br>

另外模型的整体和最后12个月的MSE和1-MAPE结果分别为：
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/SVR_img.jpg)<br>
分析:<br>
通过调节C和gamma，可以看到模型大致拟合了曲线，但是在验证集中SVR模型拟合的曲线有很强的滞后性，即单期滞后性，MSE值比较大的原因是我们对原始的Y值进行了log对数操作，并且其本身数值也比较大，指标检测的值都是基于还原的值进行的，难免MSE会很大，但模型拟合的MAPE差别也较大，从整体看来，传统的统计学习方法并不能很好的拟合曲线，准备使用深度学习LSTM方法
