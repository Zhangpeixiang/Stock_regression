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
    函数输入为降维后的【176， 24】维度的数据，将其转换为【176，3，24】，即将每三个月的数据融合在一起
    为了满足转换为3维度，转换为变为174维度，故y的标签也应该剔除前两个，从第三个开始
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
    res_data = np.array(res_list).reshape(174, 3, 24)
    return res_data
```
相信大家到目前位置已经理解了LSTM的输入过程，剩下的跟之前SVR实验部分类似：<br>
关于lstm，我们尝试用三种方式进行建模，包括设置timestep为1和3，以及验证集的划分为随机抽取和固定最后12个月这三种情况，对比一下区别<br>
M1：将timestep设置为1，并固定最后12个月为验证集，即采用如下代码处理输入数据：<br>
```python
    elif split_type == 1 and mode == 'lstm':
        # 将输入数据转换为LSTM的1个时间输入，即用之前的历史三个月的股价以及其参数预测下一个月的月度股价
        lstm_train_x = np.array(pca_descending).reshape(176, 1, 24)
        lstm_train_y = scaler_y_data
        X_train = lstm_train_x[:-12]
        X_test = lstm_train_x[-12:]
        Y_train = lstm_train_y[:-12]
        Y_test = lstm_train_y[-12:]
```
得到的数据集最终图像如下：<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/temp1.jpg)<br>
接下来我们开始进行训练，训练得到的loss图像如下：<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/train_loss.jpg)<br>
之后我们采用plot_test代码对训练得到的loss较低的模型进行具体鉴别，<br>
```python
def plot_test(train_x, org_y):
    # 记录每个loss较低的模型名称32-0.00100_3T_LSTM_model_191024SGD 34-0.00081_3T_LSTM_model_191024SGD
    model = load_model('./LSTM_3T_model/32-0.00100_3T_LSTM_model_191024SGD.h5', 'SGD')
    plt.plot(np.array(org_y), 'blue', label='true data')
    com_list = model.predict(train_x).tolist()
    # 将Z-score标准化后的数据进行还原,这里无论是预测的y还是真实的y均*std+mean就可以了，这两个需要还原为原始值
    # res_list = reshape_org_shape(des_y, com_list, org_y)
    # 将log标准化后的数据进行还原,将所有预测的y值变成以10为底的幂指数
    res_list = [math.pow(10, float(i[0])) for i in com_list]
    validation_pre = res_list[-12:]
    validation_true = np.array(org_y)[-12:]
    plt.plot(res_list, 'r', label='predict')
    plt.title('Total curve')
    plt.xlabel('Time')
    plt.ylabel('Close')
    plt.legend()
    plt.show()
    # 写入文件
    # from openpyxl import load_workbook
    # wb = load_workbook('./stock_reg_train_data.xlsx')
    # ws = wb.get_sheet_by_name('Sheet1')
    # for index, v in enumerate(res_list):
    #     ws.cell(row=2+index, column=1, value=v)
    # wb.save('./stock_reg_train_data.xlsx')
    mse = mean_squared_error(np.array(org_y), res_list)
    v_mse = mean_squared_error(validation_true, validation_pre)
    print('Total MSE', mse)
    print('Validation MSE', v_mse)
    mape_s = mape(np.array(org_y), np.array(res_list))
    v_mape = mape(validation_true, np.array(validation_pre))
    print('Total MAPE', 100-mape_s)
    print('Validation MAPE', 100-v_mape)
    validation_plot(validation_pre, validation_true)
```
如果最后程序运行没问题，应该显示的图如下所示:<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/res.jpg)<br>
测试了loss最低的模型结果如下所示:<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/8000.png)<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/test8000.png)<br>
分析:<br>
可以看到我们通过设置最后12个月作为验证集，并采用timestep为1的LSTM模型进行训练得到的最后结果如上图，模型在训练集上很好的拟合了趋势，但是对于验证集由正转负的方向可以捕捉到，但是由负转正的趋势并没有捕捉到，整体还是有一定的滞后性，但是相比于SVR模型，LSTM无论在训练集或者在测试集都更好的拟合了数据，论证了深度学习方法的有效性<br>
2. 经过以上研究我们发现了采用1个时间步长并没有重复利用lstm的性质，下面开始实验3和时间步长和6个时间步长<br>
首先看下改成3个时间步长的训练结果，训练集还是使用最后12个月之前的数据，最后12个月作为验证集，LSTM中的优化器选用SGD，训练10000次的loss图如下<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/3T_loss.jpg)<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/8000_3T_total.png)<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/8000_3T_test.png)<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/8000_3T_res.jpg)<br>
分析：<br>
将time_step改为3个时间步伐时，模型收敛的也很快，观察整体拟合情况，比之前timestep为1的时候拟合的更好，反观验证集，完全消除了滞后一期的影响，令人惊喜的发现，前三期，竟然完美的拟合了曲线，整体去年一年的验证集，12个月的数据，仅仅有3个月预测方向出现了偏差，这个模型整体的效果不得不说很强，强的惊人，并且，其中高维下滑，下滑后的回涨也完美的预测到了，比较惊人，接下来开始研究一下timestep为6的情况<br>
将timestep设置为6，同样采用后12个月作为验证集，SGD为LSTM优化器，训练10000次，观察模型效果如下：<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/6T_loss.png)<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/test_6T.png)<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/total_6T.png)<br>
不难看出，将timestep设置为半年并没有很好的提升模型的性能，反而使得模型有些过拟合，在验证集中表现出较差表现，综上，目前测试效果最好的模型是timestep为3，用最后12个月作为验证集的模型，但在实际应用中，我有一个大胆的想法，既然验证集最开始拟合的很靠谱，是不是说了训练集越长，模型越容易捕捉到其趋势？考虑到这里，获取可以尝试训练一个全部训练集的数据，然后通过实际得未来6个月的数据与有验证集的额模型进行对比，看到底是哪个在实际表现中更好<br>
3. 使用全部训练集进行训练，对比未来结果，因为这个目前还没有数据，并且可以基于之前代码进行略微修改即可，故先不实验<br>
## 实验结果分析
综上，不难看出采用LSTM，并用3期作为时间步长效果最好，我们将提取所有预测数据，进行建模作为整理结果，并预测当月的沪深300月度股价,具体得数据请参考xlsx中的temp sheet<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/2019.10plot.png)<br>
结论：
上面两幅图，第一张是整体的拟合曲线图，可以看到我们的模型对于训练集数据拟合程度在99%以上，而对于验证集数据，虽然拟合程度不高，但是12个月，仅有3个月方向预测失败，方向的准确率有67%的准确率，具体的跌转涨、涨转跌均准确预测，并没有数据滞后的感觉，整理来看是目前最优的模型，预测沪深300在10月有略微涨幅，今天是10月25日17：50，距离10月结束还有4个交易日，目前沪深300本月涨幅情况如下图<br>
![Image text](https://github.com/Zhangpeixiang/Stock_regression/blob/master/default_img/10_pre.jpg)<br>
目前来说方向应该没有问题，大概率这个月还是上涨的，但是具体涨幅是1%以内还是3~5%得看接下来这四个交易日得情况了<br>
## 可以采用NAS（神经网络结构搜索）来重构LSTM结构，训练最优的结构
## 采用对抗学习来进行训练，LSTM作为拟合器，CNN作为鉴别器进行训练
