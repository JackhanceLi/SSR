import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as scio
import random
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
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

    dense = Flatten(name='flatten')(block)
    # dense = Dense(D1)(dense)
    # dense = Activation('relu')(dense)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(dense)
    softmax = Activation('softmax', name='softmax')(dense)

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


def training(file_name, chan_num):
    # file_name: 文件名称；chan_num：通道数目
    chan_num = int(chan_num)
    raw_data = []
    # 将txt文件转换为array
    with open(file_name, 'r') as f:
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
    data = np.zeros((chan_num, raw_data.shape[1]))
    if chan_num == 4:
        data[0] = raw_data[0]
        data[1] = raw_data[1]
        data[2] = raw_data[2]
        data[3] = raw_data[3]
    elif chan_num == 5:
        data[0] = raw_data[0]
        data[1] = raw_data[1]
        data[2] = raw_data[2]
        data[3] = raw_data[3]
        data[4] = raw_data[4]
    # 得到label
    label_row = raw_data[-1].copy()

    # # 信号预处理
    fs = 1000  # Sample frequency (Hz)
    # 1、带通滤波
    f_low = 50  # low frequency
    f_high = 300  # high frequency
    data = butter_bandpass_filter(data, f_low, f_high, fs)
    # 2、循环陷波滤波
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

    clf = []  # 保存一些参数及分类模型

    classes = [str(int(i)) for i in list(set(label_row))[1:-1]]  # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    n_classes = len(classes)
    clf.append(n_classes)

    # 计算放松态均方根
    stp_rest = np.where(label_row == 11)[0][0]  # 放松态的起始时间点（start time point 起始时间点）
    rms_rest = 0
    # 计算所有通道放松数据RMS之和（10s放松数据）
    for chan in range(chan_num):
        rms_rest += np.sqrt(np.sum(data[chan, stp_rest + 3 * fs:stp_rest + 13 * fs] ** 2) / (10 * fs))
    clf.append(rms_rest)  # 保存放松态rms

    # # 根据运动数据计算用于判断运动状态的rms放大系数amplification factor
    # # 1、计算所有运动epoch的rms
    # rms_all_move = np.zeros(len(epoch_data))
    # for i in range(len(epoch_data)):
    #     for chan in range(chan_num):
    #         rms_all_move[i] += np.sqrt(np.sum(epoch_data[i, chan, 1 * fs:int(1.5 * fs)]**2)/(0.5 * fs))
    # # 2、第5个最小的rms除以放松态rms再取半作为放大系数
    # af = np.sort(rms_all_move)[4] / rms_rest / 2  # 放大系数
    af = 2  # 默认放大系数
    clf.append(af)  # 保存放大系数

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

    # # 特征提取（多通道RMS特征）
    trials_per_cla = int(len(epoch_data) / n_classes)
    f_num = 50  # 一个试次一条通道的特征数量
    clf.append(f_num)
    len_move = int(0.05 * fs)  # 帧长（用于计算rms的运动数据长度）
    clf.append(len_move)
    gap = int(0.025 * fs)  # 帧移（每隔gap ms平移一次）
    clf.append(gap)
    stp_gap = -int(0.1 * fs)  # 开始时点
    clf.append(stp_gap)
    stp = 0  # start point index
    onset = np.array(onset)

    f_rms_tr = np.zeros((epoch_data.shape[0], chan_num, f_num))
    rms_max = np.zeros(chan_num)  # 用于每个通道的归一化（非实际归一化，选择了比最大值稍小的值，排除异常值干扰）
    for ch in range(chan_num):
        rms_max[ch] = np.mean(np.sort(np.abs(epoch_data[:, ch, int(0.5 * fs):int(1.5 * fs)]).ravel())[-int(0.25 * fs)])
    clf.append(rms_max)  # 保存归一化值
    # 通道归一化
    for chan in range(chan_num):
        epoch_data[:, chan] = epoch_data[:, chan] / rms_max[chan]
    stable_stp = int(0.75 * fs)  # 固定的起始点（以防有些epoch检测不到运动起始时间）
    # clf.append(stable_stp)
    # 计算rms
    for epo in range(epoch_data.shape[0]):
        for chan in range(chan_num):
            for fnum in range(f_num):
                if not onset[epo]:
                    onset[epo] = [stable_stp]  # 固定起始点
                f_rms_tr[epo, chan, fnum] = np.sqrt(np.sum(epoch_data[epo, chan,
                                                           onset[epo][stp]+stp_gap+fnum*gap:
                                                           onset[epo][stp]+stp_gap+fnum*gap+len_move] ** 2)/len_move)

    # # 特征分类
    # 卷积神经网络CNN
    X_train = f_rms_tr.copy()
    Y_train = labels.copy()
    Y_train[np.where(Y_train == 10)] = 0

    # # 随机划分训练集验证集
    # X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.25)
    # # 前部分训练 后部分测试
    tr_index = np.arange(0, int(trials_per_cla * 0.75))
    te_index = list(set(list(np.arange(trials_per_cla))) - set(tr_index))
    tr, te = [], []
    for j in range(10):
        tr.extend(np.where(Y_train == j)[0][tr_index])
        te.extend(np.where(Y_train == j)[0][te_index])
    X_validate = X_train[tuple([te])].copy()
    Y_validate = Y_train[tuple([te])].copy()
    X_train = X_train[tuple([tr])].copy()
    Y_train = Y_train[tuple([tr])].copy()

    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train, n_classes)
    Y_validate = np_utils.to_categorical(Y_validate, n_classes)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], 1)

    kernLength, C1, D, D1 = 5, 64, 8, 64
    clf.append(kernLength)
    clf.append(C1)
    clf.append(D)
    clf.append(D1)
    model = cnn_best(nb_classes=n_classes, Chans=X_train.shape[1], Samples=X_train.shape[2],
                     dropoutRate=0.5, kernLength=kernLength, C1=C1, D=D, C2=32, D1=D1)
    print(model.summary())

    # compile the model and set the optimizers
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(X_train, Y_train, batch_size=int(X_train.shape[0] * 0.2), epochs=1000,
                        verbose=2, validation_data=(X_validate, Y_validate),
                        callbacks=[callback])
    model_path = 'online_' + str(chan_num) + 'chans_'
    joblib.dump(clf, model_path+'clf.m')  # 保存上述参数
    model.save(model_path + 'model.h5')

    print('***************************** 训练结束 *****************************')


if __name__ == '__main__':
    # 4chan-20trials/classes:20221103T160037  5chan-60trials/classes:20221118T170945
    # 5chan-45trials/classes:20221121T161151 20221122T173103  5chan-40trials/classes:20221125T173457
    file_name_ = './data/20221125T173457.txt'
    chan_num_ = 5
    training(file_name_, chan_num_)
