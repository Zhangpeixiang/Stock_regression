# Stock_regression
## 深度学习在股价中的研究 
---
前一阵研究股价预测，有些许心得，在这里主要跟大家分享下我对于LSTM回归的理解，如果有错误，欢迎批评指正<br>
研究主要分为三部分<br>
1. 传统的统计学习回归（SVR回归）<br>
支持向量机回归应该算得上是比较传统的回归模型了，当然，对于股价这种类时间序列问题，也可以采用ARIMA回归等时间序列的处理方法，在这个阶段我会一一实现这些方法
2. 主流的循环神经网络回归（LSTM）
3. 深度强化学习（实时调参）
---
## 实验研究
1. 传统统计学习回归模型以及时间序列模型<br>
先用比较经典的SVR支持向量机回归进行回归拟合，选取如下指标，采用PCA进行降维，降维后划分训练集和验证集，进行训练调参后，最后画出整体和验证集拟合曲线：<br>

其中输入的数据通过excel中的wind函数进行提取，选取了如下一些基本面指标、wind的所有技术指标、宏观指标（由于宏观指标的滞后性，有些滞后1期，有些滞后2期）：<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/org_data.jpg)<br>
通过将原始的73维数据使用PCA降维到23维，然后划分最后12个月为验证集，其余为训练集，设置C为100，gamma为0.0001利用SVR进行训练得到对比曲线图如下：<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/SVR_regression.jpg)<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/validation_img.jpg)<br>

另外模型的整体和最后12个月的MSE和1-MAPE结果分别为：
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/SVR_img.jpg)<br>
分析:<br>
通过调节C和gamma，可以看到模型大致拟合了曲线，但是在验证集中SVR模型拟合的曲线有很强的滞后性，即单期滞后性，MSE值比较大的原因是我们对原始的Y值进行了log对数操作，并且其本身数值也比较大，指标检测的值都是基于还原的值进行的，难免MSE会很大，但模型拟合的MAPE差别也较大，从整体看来，传统的统计学习方法并不能很好的拟合曲线，准备使用深度学习LSTM方法

2. 深度学习中的循环神经网络

LSTM 进行分类比较多，尤其在情感分析中，之前我的project中也讲到过，具体不做阐述，简单说下我对LSTM回归的理解，LSTM其实是采用相邻几个时间步伐的数据来对下一时刻进行预测，通俗来看，假设我们的数据格式为200行，75列，其中200行就是200个月的月度数据，75列是每个月的相关的输入数据维度`x1...x75`，LSTM的输入其实需要把`【200，75】`的数据转换为`【batchsize, timestep, dimension】`即假设一个batchsize中的一个样本的维度为`【1，3，75】`，这样可能大家就理解了，`一般网上的教程嫌弃麻烦，`在转换LSTM输入的时候，直接用np.reshape将`【200，75】`转为`【200，1，75】`这其实就是用上一个时刻的值，来预测下一个时刻，但是压根没有捕捉到LSTM的精髓，LSTM精髓就在于timestep，你设置为1，那每一个batchsize的一个样本里，只有上个月的数据，来预测下个月的数据，【1，1，75】没有用，模型会拟合的很像，为什么，`因为其实就是将模型轨迹滞后一期进行拟合，你都不如直接画个滞后一期的图像`通过前面对于LSTM的回归介绍，也可以参见我代码中的LSTM转换的部分<br>
```python
def reshape_train_data(des_data):
    """
    函数输入为降维后的【211， 20】维度的数据，将其转换为【209，3，20】，即将每三个月的数据融合在一起
    为了满足转换为3维度，转换为变为209维度，故y的标签也应该剔除前两个，从第三个开始
    :param des_data:
    :return:
    """
    des_list = des_data.tolist()
    res_list = []
    for index, data in enumerate(des_data):
        if index < len(des_data)-2:
            res_list.append(data)
            res_list.append(des_data[index+1])
            res_list.append(des_data[index+2])
        else:
            break
    res_data = np.array(res_list).reshape(209, 3, 20)
    return res_data
```
相信大家到目前位置已经理解了LSTM的输入过程，剩下的跟之前SVR实验部分类似：<br>
