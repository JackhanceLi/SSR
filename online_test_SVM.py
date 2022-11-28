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
# import pywt


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


def testing(raw_data, M, move_data, chan_num):
    chan_num = int(chan_num)
    move_data = np.array(move_data)
    len_move = 250
    y_pred = 0  # 预测值默认是0
    data = np.zeros((chan_num, raw_data.shape[1]))
    raw_data = np.array(raw_data)
    if chan_num == 2:
        data[0] = raw_data[1] - raw_data[0]
        data[1] = raw_data[3] - raw_data[2]
    elif chan_num == 4:
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
    label = int(raw_data[-1].copy()[0])
    # 信号预处理
    fs = 1000  # Sample frequency (Hz)
    # # 带通滤波
    f_low = 50  # low frequency
    f_high = 300  # high frequency
    data = butter_bandpass_filter(data, f_low, f_high, fs)
    # # 循环陷波滤波
    for i in range(5):
        f0 = 50 * (i + 1)  # Frequency to be removed from signal (Hz)
        data = notch_filter(data, f0, fs)

    # ************************************************** 画原始波形图 **************************************************
    # emg_plot(data)
    # plt.show()

    # 在线测试————实时判断开始时刻
    clf = joblib.load('./online_'+str(chan_num)+'chans_clf.m')
    # 基线均方根
    rms_rest = clf[0]
    af = clf[1]
    # 实时rms
    rms_rt = 0
    for chan in range(chan_num):
        rms_rt += np.sqrt(np.sum(data[chan] ** 2) / len_move)

    # 实时判断
    # if rms_rt > af * rms_rest and M == 3:
    if M == 3:
        move_data = np.concatenate((move_data, data.copy()), axis=1)  # 共1250ms数据
        # 识别运动开始时刻，并准确切分数据
        gap = int(0.025 * fs)  # 每隔Xms检测实时rms
        for k in range(int(move_data.shape[1] / gap)):
            rms_rt = 0
            for chan in range(chan_num):
                rms_rt += np.sqrt(np.sum(move_data[chan, k * gap:(k + 1) * gap] ** 2) / gap)  # 实时rms
            if rms_rt > af * rms_rest:
                onset = k * gap
                break

        # # 特征提取(两个通道的RMS特征)
        f_num = clf[2]  # 一个试次一条通道的特征数量
        len_move = clf[3]  # 用于计算rms的运动数据长度
        gap = clf[4]  # 每隔x ms
        stp_gap = clf[5]  # 开始时点
        rms_max = clf[6]
        f_rms_te = np.zeros((1, chan_num * f_num))
        for chan in range(chan_num):
            move_data[chan] = move_data[chan] / rms_max[chan]  # 归一化各自通道的rms
        for fnum in range(f_num):
            for chan in range(chan_num):
                f_rms_te[0, chan + fnum * chan_num] = np.sqrt(np.sum(move_data[chan,
                                                                     onset + stp_gap + fnum * gap:
                                                                     onset + stp_gap + fnum * gap + len_move] ** 2) / len_move)

        # # SVM特征分类
        svc = clf[7]
        # 得到预测值
        y_pred = svc.predict(f_rms_te)  # csp_te TransX_te
        print('y_pred: ', y_pred[0])
        print('label: ', label)

        # 将预测值与标签进行对比
        if y_pred == label:
            print('correct!')
        else:
            print('wrong~')
        # move_data = move_data[:, len_move:].copy()
        move_data = move_data[:, -1:].copy()
        M = -1

        # tcpip发送指令

    # elif rms_rt > af * rms_rest and M == 2:
    # elif M == 2:
    #     move_data = np.concatenate((move_data, data.copy()), axis=1)  # 共1000ms数据
    #     M = 3
    # elif rms_rt > af * rms_rest and M == 1:
    elif M == 1:
        move_data = np.concatenate((move_data, data.copy()), axis=1)  # 共750ms数据
        M = 3
    elif rms_rt > af * rms_rest and M == 0:
        move_data = np.concatenate((move_data, data.copy()), axis=1)  # 共500ms数据
        M = 1
    elif rms_rt < af * rms_rest:
        move_data = data.copy()
        M = 0

    return [M, move_data, y_pred]


if __name__ == '__main__':
    # import csv
    # csv_reader = csv.reader(open("D:/Brainco/Stark2代/sample/action12-0.csv"))
    # for line in csv_reader:
    #     print(line)

    # 4chan-20trials/classes:20221103T160037  5chan-60trials/classes:20221118T170945 20221121T161151 20221122T173103
    file_name = '20221122T173103.txt'
    raw_data_ = []
    with open('./data/'+file_name, 'r') as f:
        data_lists = f.readlines()
        for data_list in data_lists:
            data1 = data_list.strip('\n')  # 去掉开头和结尾的换行符
            data1 = data1.split(',')  # 把','作为间隔符
            data1 = list(map(float, data1))  # 将list中的string转换为int
            raw_data_.append(data1)
        raw_data_ = np.array(raw_data_).T
    # 画图
    # emg_plot(raw_data)
    # plt.show()
    len_move_ = 250
    move_data_ = np.zeros((1, 1))
    M_ = 0  # 运动状态 0：静息；1：检测到运动；2：运动确认；3：开始保存数据
    chan_num_ = 5
    for block in range(int(raw_data_.shape[1]/len_move_)):
        raw_data1 = raw_data_[:, len_move_*block:len_move_*(block+1)].copy()
        [M_, move_data_, y_pred_] = testing(raw_data1, M_, move_data_, chan_num_)
