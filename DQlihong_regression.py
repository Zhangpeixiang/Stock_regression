"""
author：Zhang peixiang
SVR、LSTM对沪深300指数进行日度预测
"""
from predict_each_m_v.classify import preprocess_handle
from sklearn import preprocessing
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import Callback, ModelCheckpoint


def scaler_adjust(data, scaler_model):
    if scaler_model == 1:
        # 归一化到[0,1]区间
        # x = data[]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data_nor = min_max_scaler.fit_transform(data)
        # print(x_nor[-1])
        return data_nor
    elif scaler_model == 2:
        # 归一化到[-1, 1]区间
        df = pd.DataFrame(data)
        # print(df.describe())
        # print(df)
        # df_norm = (df - df.min()) / (df.max() - df.min())
        # print(df_norm)
        df_norm = (2*(df - df.min())/(df.max() - df.min()))-1
        # print(df_norm)
        return np.array(df_norm)
    elif scaler_model == 3:
        # 归一化到[-1, 0]区间
        df = pd.DataFrame(data)
        # print(df.describe())
        # print(df)
        # df_norm = (df - df.min()) / (df.max() - df.min())
        # print(df_norm)
        df_norm = (1*(df - df.min())/(df.max() - df.min()))-1
        # print(df_norm)
        return np.array(df_norm)
    elif scaler_model == 4:
        # Z-score成标准正态
        df = pd.DataFrame(data)
        df_norm = (df - df.mean()) / (df.std())
        # print(df_norm)
        return np.array(df_norm)
    else:
        return np.array(data)

def reshape_train_data(des_data, num):
    """
    假设针对3个时间步长：
    函数输入为降维后的【176， 24】维度的数据，将其转换为【176，3，24】，即将每三个月的数据融合在一起
    为了满足转换为3维度，转换为变为174维度，故y的标签也应该剔除前两个，从第三个开始
    :param des_data:
    :return:
    """
    des_list = des_data.tolist()
    res_list = []
    for index, data in enumerate(des_data):
        if index < len(des_data)-num+1:
            for i in range(num):
                res_list.append(data+i)
        else:
            break
    res_data = np.array(res_list).reshape(176-num+1, num, 24)
    return res_data

def preprocess_handle(pca_descending, scaler_y_data, split_type, mode):
    # 加载样本数据集
    if split_type == 1 and mode == 'svr':
        X_train, X_test, Y_train, Y_test = train_test_split(pca_descending, scaler_y_data, test_size=0.2, random_state=1234565)  # 数据集分割
        return X_train, X_test, Y_train, Y_test, pca_descending, scaler_y_data
    elif split_type == 2 and mode == 'svr':
        X_train = pca_descending[:-12]
        X_test = pca_descending[-12:]
        Y_train = scaler_y_data[:-12]
        Y_test = scaler_y_data[-12:]
        return X_train, X_test, Y_train, Y_test, pca_descending, scaler_y_data
    elif split_type == 1 and mode == 'lstm':
        # 将输入数据转换为LSTM的1个时间输入，即用之前的历史三个月的股价以及其参数预测下一个月的月度股价
        lstm_train_x = np.array(pca_descending).reshape(176, 1, 24)
        lstm_train_y = scaler_y_data
        X_train = lstm_train_x[:-12]
        X_test = lstm_train_x[-12:]
        Y_train = lstm_train_y[:-12]
        Y_test = lstm_train_y[-12:]
        return X_train, X_test, Y_train, Y_test, lstm_train_x, lstm_train_y
    elif split_type == 3 and mode == 'lstm':
        # 将输入数据转换为LSTM的三个时间输入，即用之前的历史三个月的股价以及其参数预测下一个月的月度股价
        lstm_train_x = reshape_train_data(pca_descending, split_type)
        lstm_train_y = scaler_y_data[2:]
        X_train = lstm_train_x[:-12]
        X_test = lstm_train_x[-12:]
        Y_train = lstm_train_y[:-12]
        Y_test = lstm_train_y[-12:]
        return X_train, X_test, Y_train, Y_test, lstm_train_x, lstm_train_y
    elif split_type == 6 and mode == 'lstm':
        # 将输入数据转换为LSTM的三个时间输入，即用之前的历史三个月的股价以及其参数预测下一个月的月度股价
        lstm_train_x = reshape_train_data(pca_descending, split_type)
        lstm_train_y = scaler_y_data[5:]
        X_train = lstm_train_x[:-12]
        X_test = lstm_train_x[-12:]
        Y_train = lstm_train_y[:-12]
        Y_test = lstm_train_y[-12:]
        return X_train, X_test, Y_train, Y_test, lstm_train_x, lstm_train_y
    else:
        print('Please send train and test set split type model')
        X_train = [[]]
        X_test = [[]]
        Y_train = [[]]
        Y_test = [[]]
        x = [[]]
        y = [[]]
        return X_train, X_test, Y_train, Y_test, x, y

def load_data(scaler_model):
    df = pd.read_excel('./regression_data.xlsx', sheet_name='Sheet1')  # 可以通过sheet_name来指定读取的表单
    # 因为沪深3000股价采用前复权方式获取，并且其从1000到3000跨度加大，需使用log对其进行归一化
    data_nor = scaler_adjust(df.iloc[:, 1:73], scaler_model)
    # 观察其线性相关矩阵
    # df_cor = pd.DataFrame(data_nor).corr()
    # import seaborn as sns
    # sns.heatmap(df_cor, vmax=0.8, cmap='RdBu_r', square=True, center=0)
    # plt.savefig('./BluesStateRelation.png')
    # plt.show()
    # 经过标准化处理后数据变为np.array形式
    x_nor = np.array(data_nor)
    y_nor = np.log10(np.array(df.iloc[:, -1]))
    return x_nor, y_nor, df['沪深300真实值']

def descending_dimension(x):
    # 原始73维度
    # 成分占比0.99 降维到38
    # 成分占比0.95 降维到23
    # 成分占比0.90 降维到16
    pca = PCA(n_components=0.95)  # 保证降维后的数据保持95%的信息
    pca_x = pca.fit_transform(x)
    # debug后提取最后三行
    # print(pca_x[-3:, :])
    return pca_x

def validation_plot(validation_true, validation_pred):
    plt.plot(np.array(validation_true), 'blue', label='True value')
    plt.plot(np.array(validation_pred), 'r', label='Predict value')
    plt.title('Validation curve')
    plt.xlabel('Time')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

def svr_base_model(train_x, test_x, train_y, test_y, pca_x, org_y):
    # 测算MSE或者1-MAPE这两个指标均采用原始值（即log之前的y值）来进行测算
    regressor = SVR(kernel='rbf', C=10, gamma=0.00001)
    regressor.fit(train_x, train_y)
    pred_y = regressor.predict(test_x)
    pre_y = regressor.predict(pca_x)
    # 将预测出来的y值进行反log，即转换为log之前的值
    re_log_pred_y = [math.pow(10, float(i)) for i in pred_y]
    re_log_test_y = [math.pow(10, float(i)) for i in test_y]
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    max_ = org_y.argmax()
    min_ = org_y.argmin()
    res_list = []
    # 将预测出来的y值进行反log，即转换为log之前的值
    for list_ in pre_y:
        # 归一化方式不同
        # transfer = (org_y[max_] - org_y[min_]) * list_ + org_y[min_]
        transfer = math.pow(10, float(list_))
        res_list.append(round(transfer, 4))
    # print(res_list)
    plt.plot(np.array(org_y), 'blue', label='True value')
    plt.plot(np.array(res_list), 'r', label='Predict value')
    t_mape = mape(org_y, np.array(res_list))
    mape_s = mape(np.array(re_log_test_y), np.array(re_log_pred_y))
    t_mse =  mean_squared_error(org_y, res_list)
    mse = mean_squared_error(re_log_test_y, re_log_pred_y)
    print('Total MSE',t_mse)
    print('Test MSE', mse)
    print('Total 1-MAPE', t_mape)
    print('Test 1-MAPE', mape_s)
    plt.title('Total curve')
    plt.xlabel('Time')
    plt.ylabel('Close')
    plt.legend()
    plt.show()
    validation_plot(np.array(re_log_test_y), np.array(re_log_pred_y))

def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    # 先简单的直接用单个值进行预测
    model.add(LSTM(units=256, input_shape=(None, 24), return_sequences=True))
    model.add(LSTM(units=256))
    # model.add(Embedding(max_features, embedding_size, input_length=max_sentence_length))
    model.add(Dropout(0.5))
    # model.add(Bidirectional(CuDNNLSTM(128)))
    model.add(Dense(1, activation='relu'))
    # model.add(Activation('linear'))
    # model.add(Activation('sigmoid'))
    model.compile(loss='mse', optimizer='SGD')
    model.summary()
    return model

def train_model(re_train, re_test, train_y, test_y):
    # 终于找到key problem了，LSTM的输入应该为[batchsize， 【用多少天的时间进行下一天的预测】， 变量个数]
    batch_size = 8
    epoch = 10000
    model = build_model()
    checkpoint = ModelCheckpoint(filepath='./LSTM_3T_model/{epoch:02d}-{val_loss:.5f}_3T_LSTM_model_191025SGD.h5',
                                 monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min',
                                 period=500)
    history = model.fit(re_train, train_y, batch_size=batch_size, epochs=epoch, validation_data=(re_test, test_y),
                callbacks=[checkpoint])
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    print('Model has been saved')

def load_model(model_dir, optimizer):
    model = Sequential()
    # 通过输入变量个个数调整input的参数
    model.add(LSTM(units=256, input_shape=(None, 24), return_sequences=True))
    model.add(LSTM(units=256))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    # 加载模型的名字
    model.load_weights(model_dir)
    model.compile(loss='mse', optimizer=optimizer)
    return model

def lstm_base_model(train_x, test_x, train_y, test_y):
    # 通过调节参数，观察训练后的loss图，找到训练过程中最优的模型
    train_model(train_x, test_x, train_y, test_y)

def validation_plot(v_pre, v_true):
    # 对于最后12个月的数据进行画图，对比效果
    plt.plot(v_true, 'b', label='true data')
    plt.plot(v_pre, 'r', label='predict')
    plt.title('Validation curve')
    plt.xlabel('Time')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def plot_test(train_x, org_y):
    # 记录每个loss较低的模型名称
    # 8000-0.00107_1T_LSTM_model_191024SGD.h5 5000-0.00299_3T_LSTM_model_191025SGD
    model = load_model('./LSTM_3T_model/8000-0.00296_3T_LSTM_model_191025SGD.h5', 'SGD')
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
    rmse = math.sqrt(mean_squared_error(np.array(org_y), res_list))
    v_rmse = math.sqrt(mean_squared_error(validation_true, validation_pre))
    print('Total RMSE', rmse)
    print('Validation RMSE', v_rmse)
    mape_s = mape(np.array(org_y), np.array(res_list))
    v_mape = mape(validation_true, np.array(validation_pre))
    print('Total MAPE', 100-mape_s)
    print('Validation MAPE', 100-v_mape)
    validation_plot(validation_pre, validation_true)

def regression_main():
    # scaler_model 是数据标准化的方式
    # 4 ：Z-score来对输入x进行标准化
    scaler_model = 4
    # train_split_type是训练集的分割方式与lstm的时间步长的选取，
    # 1：一种是随机抽取20%作为验证集，2：一种是固定最后的12个月作为验证集
    # 对于train_mode = 'lstm'时，train_split_type还代表了time_step
    train_split_type = 3
    x, y, org_y = load_data(scaler_model)
    # pca进行降维，目前暂时降维到23维
    pca_descending = descending_dimension(x)
    # 选择建模方式
    train_mode = 'lstm'
    if train_mode == 'svr':
        train_x, test_x, train_y, test_y, total_train_x, total_train_y = preprocess_handle(pca_descending, y, train_split_type, train_mode)
        svr_base_model(train_x, test_x, train_y, test_y, pca_descending, org_y)
    elif train_mode == 'lstm':
        train_x, test_x, train_y, test_y, total_train_x, total_train_y = preprocess_handle(pca_descending, y, train_split_type, train_mode)
        # lstm_base_model(train_x, test_x, train_y, test_y)
        # lstm 会保存模型，需要人为挑选模型，并利用plot_test对模型进行挑选,org的维度根据timestep的不同而不同
        if train_split_type == 3 or train_split_type == 6:
            plot_test(total_train_x, np.array(org_y)[train_split_type-1:])
        else:
            plot_test(total_train_x, np.array(org_y))
    else:
        print('Please input Train mode right!')

if __name__ == "__main__":

    regression_main()
