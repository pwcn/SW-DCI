function filtered=ideal_passing(input,fl,fh,samplingRate)
%理想带通滤波
n=length(input);%输入数据的长度
Freq = 1:n;%等差数列
    Freq = (Freq-1)/n*samplingRate;%频率？
    mask = Freq > fl & Freq < fh;%大于fl的值变为0，小于fh的值变为1
    F = fft(input); %傅里叶变换
    F(~mask) = 0; 
    filtered = real(ifft(F));%取实部
end