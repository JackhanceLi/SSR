import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as scio
import random

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
import seaborn as sns
# import pywt
import joblib

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K


def emg_plot(data):
    y_interval = np.max(np.abs(data))
    ch_num = data.shape[0]
    # # with trigger
    # for i in range(ch_num):
    #     if i < ch_num-1:
    #         plt.plot(data[i, :]+y_interval*(ch_num-i))
    #     else:
    #         # trigger
    #         plt.plot(data[i, :]*y_interval*0.1 + y_interval * (ch_num - i))

    # # without trigger
    for i in range(ch_num):
        plt.plot(data[i, :]+y_interval*(ch_num-i))

    plt.ylim((0, (ch_num+1)*y_interval))
    # plt.xlabel('time')
    # plt.ylabel('channels (interval/'+str(int(y_interval))+r'$\mu$V'+')')
    # plt.yticks([y_interval*(ch_num-0), y_interval*(ch_num-1), y_interval*(ch_num-2),
    #             y_interval*(ch_num-3), y_interval*(ch_num-4), y_interval*(ch_num-5),
    #             y_interval*(ch_num-6), y_interval*(ch_num-7), y_interval*(ch_num-8)],
    #            ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'trigger'])
    # plt.show()


def emg_plot_fft(f, data):
    y_interval = np.max(np.abs(data))
    ch_num = data.shape[0]
    for i in range(ch_num):
        plt.plot(f, data[i, :]+y_interval*(ch_num-i))

    plt.ylim((0, (ch_num+1)*y_interval))


def notch_filter(data, f0, fs):
    Q = 30  # Quality factor
    # Design notch filter
    b, a = signal.iirnotch(f0, Q, fs)
    # freq, h = signal.freqz(b, a, fs=fs)
    # plt.plot(freq, 20*np.log10(abs(h)))
    # plt.show()
    # filtering
    y = signal.filtfilt(b, a, data)

    # plt.subplot(2,2,1)
    # emg_plot(data)
    # plt.subplot(2, 2, 2)
    # N = data.shape[1]  # 样本点个数
    # X = np.fft.fft(data)
    # X_mag = np.abs(X) / N  # 幅值除以N倍
    # f_plot = np.arange(int(N / 2 + 1))  # 取一半区间
    # X_mag_plot = 2 * X_mag[:, 0:int(N / 2 + 1)]  # 取一半区间
    # X_mag_plot[:, 0] = X_mag_plot[:, 0] / 2  # Note: DC component does not need to multiply by 2
    # emg_plot_fft(fs*f_plot/N, X_mag_plot)
    # plt.subplot(2,2,3)
    # emg_plot(y)
    # plt.subplot(2, 2, 4)
    # X_ = np.fft.fft(y)
    # X_mag_ = np.abs(X_) / N
    # X_mag_plot_ = 2 * X_mag_[:, 0:int(N / 2 + 1)]  # 取一半区间
    # X_mag_plot_[:, 0] = X_mag_plot_[:, 0] / 2  # Note: DC component does not need to multiply by 2
    # emg_plot_fft(fs*f_plot/N, X_mag_plot_)
    # plt.show()

    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = fs * 0.5  # 奈奎斯特采样频率
    low, high = lowcut / nyq, highcut / nyq
    # # 滤波器的分子（b）和分母（a）多项式系数向量
    # [b, a] = signal.butter(order, [low, high], analog=False, btype='band', output='ba')
    # # plot frequency response
    # w, h = signal.freqz(b, a, worN=2000)
    # plt.plot(w, abs(h), label="order = %d" % order)
    # # # 滤波器的二阶截面表示
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    # # # plot frequency response
    # w, h = signal.freqz(sos, worN=2000)
    # plt.plot(w, abs(h), label="order = %d" % order)
    # filtering
    y = signal.sosfiltfilt(sos, data)

    # plt.subplot(2, 2, 1)
    # emg_plot(data)
    # plt.subplot(2, 2, 2)
    # N = data.shape[1]  # 样本点个数
    # X = np.fft.fft(data)
    # X_mag = np.abs(X) / N  # 幅值除以N倍
    # f_plot = np.arange(int(N / 2 + 1))  # 取一半区间
    # X_mag_plot = 2 * X_mag[:, 0:int(N / 2 + 1)]  # 取一半区间
    # X_mag_plot[:, 0] = X_mag_plot[:, 0] / 2  # Note: DC component does not need to multiply by 2
    # emg_plot_fft(fs * f_plot / N, X_mag_plot)
    # plt.subplot(2, 2, 3)
    # emg_plot(y)
    # plt.subplot(2, 2, 4)
    # X_ = np.fft.fft(y)
    # X_mag_ = np.abs(X_) / N
    # X_mag_plot_ = 2 * X_mag_[:, 0:int(N / 2 + 1)]  # 取一半区间
    # X_mag_plot_[:, 0] = X_mag_plot_[:, 0] / 2  # Note: DC component does not need to multiply by 2
    # emg_plot_fft(fs * f_plot / N, X_mag_plot_)
    # plt.show()

    return y


def cnn(nb_classes, Chans=5, Samples=40, dropoutRate=0.5, kernLength=5, C1=16, D=8, C2=16, D1=256, norm_rate=0.25):
    input = Input(shape=(Chans, Samples, 1))
    block = Conv2D(C1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False)(input)
    # block = Conv2D(C1, (Chans, kernLength), input_shape=(Chans, Samples, 1), use_bias=False)(input)
    block = BatchNormalization()(block)
    # block = Conv2D(D, (Chans, 1), use_bias=False)(block)
    # block = SeparableConv2D(C1, kernel_size=(Chans, 1), depth_multiplier=D, depthwise_constraint=max_norm(1.), use_bias=False)(block)  # , padding='same'
    block = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = AveragePooling2D((1, 2))(block)
    block = Dropout(dropoutRate)(block)
    #
    # block = SeparableConv2D(C2, (1, 10), use_bias=False, padding='same')(block)
    # block = Conv2D(C1, (1, kernLength), use_bias=False)(block)
    # block = BatchNormalization()(block)
    # block = Activation('relu')(block)
    # block = AveragePooling2D((1, 4))(block)
    # block = Dropout(dropoutRate)(block)

    dense = Flatten()(block)
    # dense = Dense(D1)(dense)
    # dense = Activation('relu')(dense)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(dense)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input, outputs=softmax)


def cnn_best(nb_classes, Chans=5, Samples=40, dropoutRate=0.5, kernLength=5, C1=16, D=8, C2=16, D1=256, norm_rate=0.25):
    input = Input(shape=(Chans, Samples, 1))
    block = Conv2D(C1, (kernLength, kernLength), input_shape=(Chans, Samples, 1))(input)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = AveragePooling2D((1, 2))(block)
    block = Dropout(dropoutRate)(block)

    dense = Flatten()(block)
    dense = Dense(nb_classes)(dense)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input, outputs=softmax)


def cnn1(nb_classes, Chans=5, Samples=40, dropoutRate=0.5, kernLength=5, C1=16, D=8, C2=16, D1=256, norm_rate=0.25):
    input = Input(shape=(Chans, Samples, 1))
    block = Conv2D(C1, (1, 3), padding='same', input_shape=(Chans, Samples, 1))(input)
    block = Conv2D(C1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1))(block)
    block = BatchNormalization()(block)
    # block = Conv2D(D, (1, kernLength))(block)
    # block = SeparableConv2D(C1, kernel_size=(Chans, 1), depth_multiplier=D, depthwise_constraint=max_norm(1.), use_bias=False)(block)  # , padding='same'
    # block = DepthwiseConv2D((Chans, 1), depth_multiplier=D, depthwise_constraint=max_norm(1.))(block)
    # block = BatchNormalization()(block)
    # block = Activation('relu')(block)
    # block = MaxPooling2D((1, 2))(block)
    block = AveragePooling2D((1, 2))(block)
    block = Dropout(dropoutRate)(block)

    dense = Flatten()(block)
    # dense = Dense(D1)(dense)
    # dense = Activation('relu')(dense)
    dense = Dense(nb_classes)(dense)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input, outputs=softmax)


if __name__ == '__main__':
    # 4chan-20trials/classes:20221103T160037  5chan-60trials/classes:20221118T170945
    # 5chan-45trials/classes:20221121T161151 20221122T173103  5chan-40trials/classes:20221125T173457
    file_name = '20221125T173457'
    raw_data = []
    # 将txt文件转换为array
    with open('./data/'+file_name+'.txt', 'r') as f:
        data_lists = f.readlines()
        for data_list in data_lists:
            data1 = data_list.strip('\n')  # 去掉开头和结尾的换行符
            data1 = data1.split(',')  # 把','作为间隔符
            data1 = list(map(float, data1))  # 将list中的string转换为int
            raw_data.append(data1)
        raw_data = np.array(raw_data).T

    # 画图
    # emg_plot(raw_data)
    # plt.show()

    # 根据通道数目提取相应肌电数据
    if file_name == '20221103T160037':
        chan_num = 4
        data = raw_data[0:4]
    else:
        chan_num = 5
        data = raw_data[0:5]
    label_row = raw_data[-1].copy()

    # # 信号预处理
    fs = 1000  # Sample frequency (Hz)
    # 1、带通滤波
    f_low = 50  # low frequency
    f_high = 300  # high frequency
    data = butter_bandpass_filter(data, f_low, f_high, fs)
    # 2、循环陷波滤波 notch (band-stop) filters were applied at 50 Hz and
    # its harmonics below the sampling rate to remove the power line noise
    for f0 in np.arange(np.ceil(f_low/50)*50, f_high+1, 50):
        # f0: Frequency to be removed from signal (Hz)
        data = notch_filter(data, int(f0), fs)
    # 3、降采样
    # fs = 500
    data = data[:, ::int(1000 / fs)]
    label_row = label_row[::int(1000 / fs)]

    # ************************************************ 画滤波后原始波形图 ************************************************
    # # 单图
    # emg_plot(data)
    # plt.show()
    # # 子图
    # fig, ax = plt.subplots(len(data))
    # y_interval = np.ceil(np.max(np.abs(data))/100)*100
    # for ij in range(len(data)):
    #     ax[ij].plot(data[ij, :])
    #     ax[ij].set_ylim(-y_interval, y_interval)
    # plt.show()

    # # 删除某一些label数据
    # for i in [10, 1]:
    #     label_row[np.where(label_row == i)] = 0
    classes = [str(int(i)) for i in list(set(label_row))[1:-1]]  # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    n_classes = len(classes)

    # 按label大致分割数据
    epoch_data, labels = [], []
    index = 0
    flag = 0
    epoch_length = int(2.5 * fs)  # 每个epoch的数据长度（2.5s）
    while index < len(label_row):
        # label=0是休息，label=11表示放松态肌电数据，label=1-10分别表示1234567890
        if label_row[index] != 0 and label_row[index] != 11:
            epoch_data.append(data[:, index:index+epoch_length])
            labels.append(label_row[index])
            # ******以下代码画图用******
            if flag == 0:
                epoch_data_ = data[:, index:index+epoch_length].copy()
                flag = 1
            else:
                epoch_data_ = np.concatenate((epoch_data_, data[:, index:index+epoch_length].copy()), axis=1)
            # ******以上代码画图用******
            # index往后移epoch_length个点
            index += epoch_length
        index += 1
    epoch_data = np.array(epoch_data)
    labels = np.array(labels).astype(int)

    # ************************************************ 画粗略分段波形图 ************************************************
    # # 单图
    # emg_plot(epoch_data_)
    # for i in range(len(epoch_data)):
    #     plt.axvline(epoch_length * i, c='k')
    # plt.show()
    # # 子图
    # fig, ax = plt.subplots(epoch_data_.shape[0])
    # y_epoch_length = np.ceil(np.max(np.abs(epoch_data_))/100)*100
    # for ij in range(epoch_data_.shape[0]):
    #     ax[ij].plot(epoch_data_[ij, :])
    #     ax[ij].set_ylim(-y_epoch_length, y_epoch_length)
    #     # ax[ij].set_xticks(range(int(epoch_length/2), epoch_data_.shape[1], epoch_length), range(0, epoch_data.shape[0]))
    #     for i in range(len(epoch_data)):
    #         ax[ij].axvline(epoch_length*i, c='k')
    # plt.tight_layout()
    # plt.show()

    # 计算放松态均方根
    stp_rest = np.where(label_row == 11)[0][0]  # 放松态的起始时间点（start time point 起始时间点）
    rms_rest = 0
    # 计算所有通道放松数据RMS之和（10s放松数据）
    for chan in range(chan_num):
        rms_rest += np.sqrt(np.sum(data[chan, stp_rest + 3 * fs:stp_rest + 13 * fs] ** 2) / (10 * fs))

    # # 根据运动数据计算用于判断运动状态的rms放大系数amplification factor
    # 1、计算所有运动epoch的rms
    # rms_all_move = np.zeros(len(epoch_data))
    # for i in range(len(epoch_data)):
    #     for chan in range(chan_num):
    #         rms_all_move[i] += np.sqrt(np.sum(epoch_data[i, chan, 1 * fs:int(1.5 * fs)]**2)/(2 * fs))
    # # 2、第5个最小的rms除以放松态rms再取半作为放大系数
    # af = np.sort(rms_all_move)[4] / rms_rest / 2  # 放大系数
    af = 2  # 默认放大系数

    # 识别运动开始时刻，并准确切分数据
    interval = int(0.025 * fs)  # 每隔 interval ms检测实时rms
    onset, onset_ = [], []  # 运动开始时刻（在每一个epoch里/为了在整条波形上画图）
    for j in range(len(epoch_data)):
        k_ = []
        for k in range(int(epoch_length/interval)):
            rms_rt = 0
            for chan in range(chan_num):
                rms_rt += np.sqrt(np.sum(epoch_data[j, chan, k*interval:(k+1)*interval]**2)/interval)  # 实时rms
            if rms_rt > af*rms_rest:
                k_.append(k*interval)
                onset_.append(k*interval+j*epoch_length)
        # 仅标注运动epoch的开始时刻在onset里
        onset.append(k_)

    # ************************************************ 画细致分段波形图 ************************************************
    # emg_plot(epoch_data_)
    # for i in range(len(onset_)):
    #     plt.axvline(onset_[i], c='k')
    # plt.show()

    # # 看一下每个类持续的时长
    # for epo in range(len(onset)):
    #     if not onset[epo]:
    #         onset[epo] = [675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 1000, 1025, 1225, 1250]  # 固定起始点
    # a = [onset[i][-1] - onset[i][0] for i in range(len(onset))]
    # a = np.array(a)
    # aa = np.zeros((10, int(len(epoch_data) / n_classes)))
    # for i in range(10):
    #     aa[i] = a[np.where(labels == i + 1)]
    # aa1 = np.average(aa, axis=1)

    # 随机划分训练集和测试集
    trials_per_cla = int(len(epoch_data) / n_classes)
    tr_num = 30  # 每个类的训练样本数 int(trials_per_cla*0.7)
    f_num = 50  # 一个试次一条通道的特征数量
    rann = 5  # 随机次数
    folds = 5  # 6:2:2交叉验证折数
    acc_fold_ran = np.zeros((folds, rann))
    CM_fold_ran = np.zeros((n_classes, n_classes))  # 画混淆矩阵用
    for fold in range(folds):
        acc_ran = np.zeros(rann)
        CM_ran = np.zeros((n_classes, n_classes))  # 画混淆矩阵用
        for ran in range(rann):
            # # 6:2:2划分训练集:验证集:测试集 每类27:9:9
            te_index = np.arange(fold*int(trials_per_cla*0.2), (fold+1)*int(trials_per_cla*0.2))
            tr_index = list(set(list(np.arange(trials_per_cla))) - set(te_index))

            # # 共15组数据，每组每类3个试次，把最后5组数据作为测试集，前面10组作为训练集
            # tr_index = np.arange(0, tr_num)
            # te_index = list(set(list(np.arange(trials_per_cla))) - set(tr_index))

            # # 随机划分训练集和测试集
            # random.seed(ran)
            # tr_index = random.sample(list(np.arange(trials_per_cla)), tr_num)
            # te_index = list(set(list(np.arange(trials_per_cla))) - set(tr_index))

            tr, te = [], []
            for j in classes:
                tr.extend(np.where(labels==int(j))[0][tr_index])
                te.extend(np.where(labels==int(j))[0][te_index])
            X_tr = epoch_data[tuple([tr])].copy()
            y_tr = labels[tuple([tr])].copy()
            X_te = epoch_data[tuple([te])].copy()
            y_te = labels[tuple([te])].copy()
            onset_tr = np.array(onset)[tuple([tr])].copy()
            onset_te = np.array(onset)[tuple([te])].copy()

            # # 特征提取（多通道RMS特征）
            len_move = int(0.05 * fs)  # 帧长（用于计算rms的运动数据长度）
            gap = int(0.025 * fs)  # 帧移（每隔gap ms平移一次）
            stp_gap = -int(0.1 * fs)  # 开始时点
            stp = 0  # start point in the onset
            stable_stp = int(0.75 * fs)  # 固定的起始点（以防有些epoch检测不到运动起始时间）

            # # 提取训练集rms特征
            f_rms_tr = np.zeros((X_tr.shape[0], chan_num, f_num))
            rms_max = np.zeros(chan_num)  # 用于每个通道的归一化（非实际归一化，选择了比最大值稍小的值，排除异常值干扰）
            for ch in range(chan_num):
                rms_max[ch] = np.mean(np.sort(np.abs(epoch_data[:, ch, int(0.5 * fs):int(1.5 * fs)]).ravel())[-int(0.25 * fs)])
            # rms_max = np.max(np.max(np.abs(X_tr), axis=2), axis=0)
            # 通道归一化
            for chan in range(chan_num):
                X_tr[:, chan] = X_tr[:, chan] / rms_max[chan]
            # 计算rms
            for epo in range(X_tr.shape[0]):
                for chan in range(chan_num):
                    for fnum in range(f_num):
                        if not onset_tr[epo]:
                            onset_tr[epo] = [stable_stp]  # 固定起始点
                        f_rms_tr[epo, chan, fnum] = np.sqrt(np.sum(X_tr[epo, chan,
                                                                        onset_tr[epo][stp]+stp_gap+fnum*gap:
                                                                        onset_tr[epo][stp]+stp_gap+fnum*gap+len_move] ** 2)/len_move)

            # # 提取测试集rms特征
            f_rms_te = np.zeros((X_te.shape[0], chan_num, f_num))
            for chan in range(chan_num):
                X_te[:, chan] = X_te[:, chan] / rms_max[chan]
            for epo in range(X_te.shape[0]):
                for chan in range(chan_num):
                    for fnum in range(f_num):
                        if not onset_te[epo]:
                            onset_te[epo] = [stable_stp]  # 固定起始点
                        f_rms_te[epo, chan, fnum] = np.sqrt(np.sum(X_te[epo, chan,
                                                                        onset_te[epo][stp]+stp_gap+fnum*gap:
                                                                        onset_te[epo][stp]+stp_gap+fnum*gap+len_move] ** 2) / len_move)

            # # 特征选择
            # # 初始化selector（多种score_func可选，此处列举两种）
            # selector = SelectKBest(score_func=mutual_info_classif, k=int(f_rms_tr.shape[1]*0.95))
            # # selector = SelectKBest(score_func=f_classif, k=k+1)
            # # 训练selector
            # selector.fit(f_rms_tr, y_tr)
            # # 得到训练集和测试集的最优特征
            # f_rms_tr = selector.transform(f_rms_tr)
            # f_rms_te = selector.transform(f_rms_te)

            # # 特征分类
            # 卷积神经网络CNN
            X_train = f_rms_tr.copy()
            X_test = f_rms_te.copy()
            Y_train = y_tr.copy()
            Y_test = y_te.copy()
            Y_train[np.where(Y_train == 10)] = 0
            Y_test[np.where(Y_test == 10)] = 0

            # # 随机划分训练集验证集
            # X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.25, random_state=ran)
            # # 前部分训练 后部分测试
            tr_index = np.arange(0, int((trials_per_cla-int(trials_per_cla*0.2))*0.75))
            te_index = list(set(list(np.arange(trials_per_cla-int(trials_per_cla*0.2)))) - set(tr_index))
            tr, te = [], []
            for j in range(10):
                tr.extend(np.where(Y_train==j)[0][tr_index])
                te.extend(np.where(Y_train==j)[0][te_index])
            X_validate = X_train[tuple([te])].copy()
            Y_validate = Y_train[tuple([te])].copy()
            X_train = X_train[tuple([tr])].copy()
            Y_train = Y_train[tuple([tr])].copy()

            # convert labels to one-hot encodings.
            Y_train = np_utils.to_categorical(Y_train, n_classes)
            Y_validate = np_utils.to_categorical(Y_validate, n_classes)
            Y_test = np_utils.to_categorical(Y_test, n_classes)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
            X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

            kernLength, C1, D, D1 = 5, 64, 8, 64
            model = cnn_best(nb_classes=n_classes, Chans=X_train.shape[1], Samples=X_train.shape[2],
                        dropoutRate=0.5, kernLength=kernLength, C1=C1, D=D, C2=32, D1=D1)
            print(model.summary())

            # compile the model and set the optimizers
            opt = Adam()
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            # count number of parameters in the model
            numParams = model.count_params()

            model_path = './tmp_CNN/'+file_name+'_'

            # set a valid path for your system to record model checkpoints
            checkpointer = ModelCheckpoint(filepath=model_path+'checkpoint.h5', verbose=1,
                                           save_best_only=True)

            callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

            history = model.fit(X_train, Y_train, batch_size=int(X_train.shape[0]*0.2), epochs=1000,
                                verbose=2, validation_data=(X_validate, Y_validate),
                                callbacks=[callback])
            model.save(model_path+'model.h5')

            # # 画图
            # acc = history.history['accuracy']
            # val_acc = history.history['val_accuracy']
            # loss = history.history['loss']
            # val_loss = history.history['val_loss']
            #
            # epochs = range(len(acc))
            #
            # plt.plot(epochs, acc, 'b', label='Training acc')
            # plt.plot(epochs, val_acc, 'r', label='Validation acc')
            # plt.title('Training and validation accuracy')
            # plt.legend()
            #
            # plt.figure()
            #
            # plt.plot(epochs, loss, 'b', label='Training loss')
            # plt.plot(epochs, val_loss, 'r', label='Validation loss')
            # plt.title('Training and validation loss')
            # plt.legend()
            #
            # plt.show()

            # testing
            model1 = cnn_best(nb_classes=n_classes, Chans=X_train.shape[1], Samples=X_train.shape[2],
                         dropoutRate=0.5, kernLength=kernLength, C1=C1, D=D, C2=32, D1=D1)
            model1.load_weights(model_path + 'model.h5')
            probs = model1.predict(X_test)
            y_pred = probs.argmax(axis=-1)
            acc = np.mean(y_pred == Y_test.argmax(axis=-1))
            print("Classification accuracy_lastmodel: %f " % (acc))
            acc_ran[ran] = acc

            # # load optimal weights
            # model.load_weights(model_path+'checkpoint.h5')
            #
            # probs = model.predict(X_test)
            # y_pred = probs.argmax(axis=-1)
            # acc = np.mean(y_pred == Y_test.argmax(axis=-1))
            # print("Classification accuracy_bestmodel: %f " % (acc))
            # acc_ran[ran] = acc

            # 画混淆矩阵
            y_pred[np.where(y_pred == 0)] = 10
            print('y_te: ', y_te)
            print('y_pred: ', y_pred)
            print('accuracy of the test set:', acc_ran[ran])
            CM = confusion_matrix(y_te, y_pred, labels=[int(i) for i in classes])
            print(CM)
            for cm in range(n_classes):
                print(classes[cm] + ':', CM[cm, cm] / np.sum(CM[cm]))
            CM_ran += confusion_matrix(y_te, y_pred, labels=[int(i) for i in classes])

        acc_ran_avg = np.average(acc_ran)
        print('acc with diff rans:')
        print(acc_ran)
        print('average accuracy across rans:', acc_ran_avg)

        # print('confusion matrix number-wise:')
        # print(CM_ran)
        # print('confusion matrix percent-wise:')
        # print(CM_ran / np.sum(CM_ran[0]))
        # print('accuracy:')
        # for ii in range(n_classes):
        #     print(CM_ran[ii, ii] / np.sum(CM_ran[ii]))
        # CM_ran_acc = np.zeros((n_classes, n_classes))
        # for cla in range(n_classes):
        #     CM_ran_acc[cla] = CM_ran[cla] / np.sum(CM_ran[cla])
        # sns.set()
        # f, ax = plt.subplots()
        # # classes[-1] = '0'
        # sns.heatmap(CM_ran_acc, annot=True, ax=ax, xticklabels=[int(i) for i in classes], yticklabels=[int(i) for i in classes])  # 画热力图
        #
        # ax.set_title('confusion matrix, acc:  CNN:' + str(acc_ran_avg))  # 标题
        # ax.set_xlabel('predict')  # x轴
        # ax.set_ylabel('true')  # y轴
        # plt.show()

        acc_fold_ran[fold] = acc_ran
        CM_fold_ran += CM_ran

    acc_fold_ran_avg = np.average(acc_fold_ran)
    print('acc with diff folds and rans:')
    print(acc_fold_ran)
    print('average accuracy in folds:', np.average(acc_fold_ran, axis=1))
    print('average accuracy across folds:', acc_fold_ran_avg)

    print('confusion matrix number-wise:')
    print(CM_fold_ran)
    print('confusion matrix percent-wise:')
    print(CM_fold_ran / np.sum(CM_fold_ran[0]))
    print('accuracy:')
    for ii in range(n_classes):
        print(CM_fold_ran[ii, ii] / np.sum(CM_fold_ran[ii]))
    CM_fold_ran_acc = np.zeros((n_classes, n_classes))
    for cla in range(n_classes):
        CM_fold_ran_acc[cla] = CM_fold_ran[cla] / np.sum(CM_fold_ran[cla])
    sns.set()
    f, ax = plt.subplots()
    classes[-1] = '0'
    sns.heatmap(CM_fold_ran_acc, annot=True, ax=ax, xticklabels=[int(i) for i in classes],
                yticklabels=[int(i) for i in classes])  # 画热力图

    ax.set_title('confusion matrix, acc:  CNN:' + str(acc_fold_ran_avg))  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()


