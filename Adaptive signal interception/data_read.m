%% 读取数据
%clc;clear;
%path = 'G:\动脉硬化\数据\指尖数据'; %想要查看的文件夹的路径
%path = 'I:\康复中心数据（2021.09.22-今）\2022.01.04采集\finger_data';
%f = fullfile(path,'Dataprocess.txt');
data = load('Dataprocess.txt');
pause_figer=data';
plot(pause_figer)
fftplot(pause_figer)
%% 数据预处理
bvp_pause=ideal_passing(pause_figer,0.8,4,200);%带通滤波
H_raw=bvp_pause;
Avg = mean(H_raw);
Sd = std(H_raw);%标准差，默认除以n，即有偏的
normal_h = (H_raw-Avg)/Sd;%归一化
figure(1)
plot(normal_h)%画图（figure1显示出来）

%信号加窗，返回峰值方差最小的三个sub_PPG,下标、最小值、数据
[m,a_min,PPG_good,S_dex,S_max,PPG_Sgood] = Windows_var(normal_h);
figure(2)
subplot(311)
plot(PPG_good(1,:))%画图（figure1显示出来）
subplot(312)
plot(PPG_good(2,:))%画图（figure1显示出来）
subplot(313)
plot(PPG_good(3,:))%画图（figure1显示出来）

%% 滤波
y_Gass = Gaussianfilter(pause_figer);%高斯滤波

% 移动平均滤波器
windowSize = 5; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
y_1D = filter(b,a,pause_figer);

y_kalman = kalmanFilter(pause_figer);

%% 数据保存

% PPGdata(1,:) = PPG_good(1,:)

% 因难以排除严重干扰的数据，为了保证数据的可靠性，人工筛选每一个病人PPG数据，
%
%%
% SNR = snr(PPG_good(1,:),200);

