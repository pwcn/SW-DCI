function filtered=ideal_passing(input,fl,fh,samplingRate)
%�����ͨ�˲�
n=length(input);%�������ݵĳ���
Freq = 1:n;%�Ȳ�����
    Freq = (Freq-1)/n*samplingRate;%Ƶ�ʣ�
    mask = Freq > fl & Freq < fh;%����fl��ֵ��Ϊ0��С��fh��ֵ��Ϊ1
    F = fft(input); %����Ҷ�任
    F(~mask) = 0; 
    filtered = real(ifft(F));%ȡʵ��
end