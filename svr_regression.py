"""
author：Zhang peixiang
SVR对沪深300指数进行日度预测
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

def preprocess_handle(pca_descending, scaler_y_data, split_type):
    # 加载样本数据集
    if split_type == 1:
        X_train, X_test, Y_train, Y_test = train_test_split(pca_descending, scaler_y_data, test_size=0.2, random_state=1234565)  # 数据集分割
    elif split_type == 2:
        X_train = pca_descending[:-12]
        X_test = pca_descending[-12:]
        Y_train = scaler_y_data[:-12]
        Y_test = scaler_y_data[-12:]
    else:
        print('Please send train and test set split type model')
        X_train = [[]]
        X_test = [[]]
        Y_train = [[]]
        Y_test = [[]]
    return X_train, X_test, Y_train, Y_test

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
    # debug here
    # print(x_nor[-1, 0:18])
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

def regression_main():
    # scaler_model 是数据标准化的方式
    # Z-score来对输入x进行标准化
    scaler_model = 4
    # train_split_type是训练集的分割方式，一种是随机抽取20%作为验证集，一种是固定最后的12个月作为验证集
    train_split_type = 2
    x, y, org_y = load_data(scaler_model)
    # pca 进行降维， 目前暂时降维到23维
    pca_descending = descending_dimension(x)
    train_x, test_x, train_y, test_y = preprocess_handle(pca_descending, y, train_split_type)
    svr_base_model(train_x, test_x, train_y, test_y, pca_descending, org_y)


if __name__ == "__main__":

    regression_main()
