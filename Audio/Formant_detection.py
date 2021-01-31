# Autor: schou
# Language: Python 3.7
# source:https://blog.csdn.net/sinat_18131557/article/details/106017598
import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import librosa
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def lpc_coeff(s,p):
    """
    :param s: 一帧数据
    :param p:线性预测的阶数
    :return:
    """
    n = len(s)
    #计算自相关函数
    Rp = np.zeros(p)
    for i in range(p):
        Rp[i] = np.sum(np.multiply(s[i+1:n],s[:n - i - 1]))
    Rp0 = np.matmul(s,s.T)
    Ep = np.zeros((p,1))
    k = np.zeros((p,1))
    a = np.zeros((p,p))

    #处理i = 0的情况
    Ep0 = Rp0
    k[0] = Rp[0] / Rp0
    a[0,0] = k[0]
    Ep[0] = (1 - k[0] *k[0])*Ep0
    #i= 1开始，递归计算
    if p > 1:
        for i in range(1,p):
            k[i] = (Rp[i] - np.sum(np.multiply(a[:i,i-1],Rp[i-1::-1])))/Ep[i-1]
            a[i,i] = k[i]
            Ep[i] = (1 - k[i] * k[i])*Ep[i-1]
            for j in range(i-1,-1,-1):
                a[j,i] = a[j,i-1] - k[i]*a[i-j-1,i-1]
    ar = np.zeros(p+1)
    ar[0] = 1
    ar[1:] = -a[:,p-1]
    G = np.sqrt(Ep[p-1])
    return ar,G

def lpcff(ar,npp = None):
    """

    :param ar: 线性预测系数
    :param npp: FFT阶数
    :return:
    """
    p1 = ar.shape[0]
    if npp is None:
        npp = p1 - 1
    ff = 1 / np.fft.fft(ar,2 * npp + 2)
    return ff[:len(ff)//2]

def lpc_lpccm(ar,n_lpc,n_lpcc):
    lpcc = np.zeros(n_lpcc)
    lpcc[0] = ar[0]  #计算n=1的lpcc
    for n in range(1,n_lpc):  #计算n = 2,,pp的lpcc
        lpcc[n] = ar[n]
        for l in range(n-1):
            lpcc[n] += ar[1] * lpcc[n-1] * (n-1) / n
    for n in range(n_lpc,n_lpcc):  #计算n > p 的lpcc
        lpcc[n] = 0
        for l in range(n_lpc):
            lpcc[n] += ar[1] * lpcc[n-1] *(n-1) / n
    return -lpcc


def local_maximum(x):
    """
    求序列极值
    :param x:
    :return:
    """
    d = np.diff(x)
    l_d = len(d)
    maxium = []
    loc = []
    for i in range(l_d-1):
        if d[i] > 0 and d[i+1] <= 0:
            maxium.append(x[i+1])
            loc.append(i+1)
    return maxium,loc

def Formant_Cepst(u,cepstl):
    """
    倒谱法共振峰估计函数
    :param u:
    :param cepstl:
    :return:
    """
    wlen2 = len(u)//2
    U = np.log(np.abs(np.fft.fft(u)[:wlen2]))
    Cepst = np.fft.ifft(U)
    cepst = np.zeros(wlen2,dtype = np.complex)
    cepst[:cepstl] = Cepst[:cepstl]
    cepst[-cepstl+1:] = cepst[-cepstl+1:]
    spec = np.real(np.fft.fft(cepst))
    val,loc = local_maximum(spec)
    return val,loc,spec

def Formant_Interpolation(u,p,fs):
    """
    插值法估计共振峰函数
    :param u:
    :param p:
    :param fs:
    :return:
    """
    ar,_ = lpc_coeff(u,p)
    #查看输出的属性
    U = np.power(np.abs(np.fft.rfft(ar,2*255)),-2)
    df = fs/512
    val,loc = local_maximum(U)
    ll = len(loc)
    pp = np.zeros(ll)
    F = np.zeros(ll)
    Bw = np.zeros(ll)
    for k in range(ll):
        m = loc[k]
        m1,m2 = m-1,m+1
        p = val[k]
        p1,p2 = U[m1],U[m2]
        aa = (p1+p2)/2 - p
        bb = (p2 -p1)/2
        cc = p
        dm = -bb/2/aa
        pp[k] = - bb * bb / 4 / aa + cc #中心频率对应的功率谱Hp
        m_new = m + dm
        bf = -np.sqrt(bb*bb -4*aa *(cc - pp[k]/2))/aa
        F[k] = (m_new-1)*df
        Bw[k] = bf * df
    return F,Bw,pp,U,loc

def Formant_Root(u,p,fs,n_frmnt):
    """
    LPC求根法的共振峰估计函数
    :param u:
    :param p:
    :param fs:
    :param n_frmnt:
    :return:
    """
    ar,_ = lpc_coeff(u,p)
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
    const = fs / (2 * np.pi)
    rts = np.roots(ar)
    yf = []
    Bw = []
    for i in range(len(ar)-1):
        re = np.real(rts[i])
        im = np.imag(rts[i])
        fromn = const * np.arctan2(im,re)
        bw = -2 * const * np.log(np.abs(rts[i]))
        if fromn > 150 and bw < 700 and fromn < fs / 2:
            yf.append(fromn)
            Bw.append(bw)
    return yf[:min(len(yf),n_frmnt)],Bw[:min(len(Bw),n_frmnt)],U


plt.figure(figsize=(14,12))
#path = "D:\Code_gitee\python_sound_open\chapter4_特征提取\C4_3_y.wav"
path = "C4_3_y.wav"
data,fs = librosa.load(path)
u = lfilter([1,-0.99],[1],data)

cepstl = 6
wlen = len(u)
wlen2 = wlen//2
#预处理-加窗
u2 = np.multiply(u,np.hamming(wlen))
#预处理-FFT，取对数
U_abs = np.log(np.abs(np.fft.fft(u2))[:wlen2])
#4.3.1
freq = [i*fs/wlen for i in range(wlen2)]
val,loc,spec = Formant_Cepst(u,cepstl)
plt.subplot(4,1,1)
plt.plot(freq,U_abs,'k')
plt.title('频谱')

plt.subplot(4,1,2)
plt.plot(freq,spec,'k')
plt.title('倒谱法共振峰估计')
for i in range(len(loc)):
    #plt.subplot(4,1,2)
    plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(spec), spec[loc[i]]], '-.k')
    plt.text(freq[loc[i]], spec[loc[i]], 'Freq={}'.format(int(freq[loc[i]])))

#4.3.2
p = 12
freq = [i * fs / 512 for i in range(256)]
F,Bw,pp,U,loc = Formant_Interpolation(u,p,fs)

plt.subplot(4,1,3)
plt.plot(freq,U)
plt.title("LPC内插法的共振峰估计")
for i in range(len(Bw)):
    #plt.subplot(4, 1, 3)
    plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(U), U[loc[i]]], '-.k')
    plt.text(freq[loc[i]], U[loc[i]], 'Freq={:.0f}\nHp={:.2f}\nBw={:.2f}'.format(F[i], pp[i], Bw[i]))

#4.3.3
# p = 12
# freq = []
n_frmnt = 4
F,Bw,U = Formant_Root(u,p,fs,n_frmnt)
plt.subplot(4,1,4)
plt.plot(freq,U)
plt.title("LPC求根法")
for i in range(len(Bw)):
    #plt.subplot(4, 1, 4)
    plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(U), U[loc[i]]], '-.k')
    plt.text(freq[loc[i]], U[loc[i]], 'Freq={:.0f}\nBw={:.2f}'.format(F[i], Bw[i]))
plt.show()
plt.savefig('images/共振峰估计.png')
plt.close()