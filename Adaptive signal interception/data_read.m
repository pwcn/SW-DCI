%% ��ȡ����
%clc;clear;
%path = 'G:\����Ӳ��\����\ָ������'; %��Ҫ�鿴���ļ��е�·��
%path = 'I:\�����������ݣ�2021.09.22-��\2022.01.04�ɼ�\finger_data';
%f = fullfile(path,'Dataprocess.txt');
data = load('Dataprocess.txt');
pause_figer=data';
plot(pause_figer)
fftplot(pause_figer)
%% ����Ԥ����
bvp_pause=ideal_passing(pause_figer,0.8,4,200);%��ͨ�˲�
H_raw=bvp_pause;
Avg = mean(H_raw);
Sd = std(H_raw);%��׼�Ĭ�ϳ���n������ƫ��
normal_h = (H_raw-Avg)/Sd;%��һ��
figure(1)
plot(normal_h)%��ͼ��figure1��ʾ������

%�źżӴ������ط�ֵ������С������sub_PPG,�±ꡢ��Сֵ������
[m,a_min,PPG_good,S_dex,S_max,PPG_Sgood] = Windows_var(normal_h);
figure(2)
subplot(311)
plot(PPG_good(1,:))%��ͼ��figure1��ʾ������
subplot(312)
plot(PPG_good(2,:))%��ͼ��figure1��ʾ������
subplot(313)
plot(PPG_good(3,:))%��ͼ��figure1��ʾ������

%% �˲�
y_Gass = Gaussianfilter(pause_figer);%��˹�˲�

% �ƶ�ƽ���˲���
windowSize = 5; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
y_1D = filter(b,a,pause_figer);

y_kalman = kalmanFilter(pause_figer);

%% ���ݱ���

% PPGdata(1,:) = PPG_good(1,:)

% �������ų����ظ��ŵ����ݣ�Ϊ�˱�֤���ݵĿɿ��ԣ��˹�ɸѡÿһ������PPG���ݣ�
%
%%
% SNR = snr(PPG_good(1,:),200);

