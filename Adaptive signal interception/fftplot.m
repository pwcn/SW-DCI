function fftplot(input)

% fs = 200;  % 重采样频率
% T = 1/fs;  % 周期
% n = 5;  % 1Hz频率被分成n段
% N = fs*n;  % 因为1Hz频率被分成了n段，所以频谱的x轴数组有fs*n个数
% f = (0: N-1)*fs/N;  % 将fs个频率细分成fs*n个（即原来是[0, 1, 2, …, fs]，现在是[0, 1/N, 2/N, …, (N-1)*fs/N]）
% t = (0: N-1)*T;  % 信号所持续的时长（N个周期）
% nHz = 100;  % 画的频谱的横坐标到nHz
% Hz = nHz*n;  % 画的频谱的横坐标的数组个数
% 
% 
% fx = abs(fft(input-mean(input)))/(N/2);  % 
% plot(f(1:Hz), fx(1:Hz),'k'),title('原始信号频域'),xlabel('frequency [Hz]'); 

Fs = 200;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = length(input);             % Length of signal

Y = fft(input);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of PPG')
xlabel('f (Hz)')
ylabel('|P1(f)|')
end