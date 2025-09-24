function fftplot(input)

% fs = 200;  % �ز���Ƶ��
% T = 1/fs;  % ����
% n = 5;  % 1HzƵ�ʱ��ֳ�n��
% N = fs*n;  % ��Ϊ1HzƵ�ʱ��ֳ���n�Σ�����Ƶ�׵�x��������fs*n����
% f = (0: N-1)*fs/N;  % ��fs��Ƶ��ϸ�ֳ�fs*n������ԭ����[0, 1, 2, ��, fs]��������[0, 1/N, 2/N, ��, (N-1)*fs/N]��
% t = (0: N-1)*T;  % �ź���������ʱ����N�����ڣ�
% nHz = 100;  % ����Ƶ�׵ĺ����굽nHz
% Hz = nHz*n;  % ����Ƶ�׵ĺ�������������
% 
% 
% fx = abs(fft(input-mean(input)))/(N/2);  % 
% plot(f(1:Hz), fx(1:Hz),'k'),title('ԭʼ�ź�Ƶ��'),xlabel('frequency [Hz]'); 

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