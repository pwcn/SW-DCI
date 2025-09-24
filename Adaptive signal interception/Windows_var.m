function [m,a_min,PPG_good,S_dex,S_max,PPG_Sgood] = Windows_var(input_data)
    len = length(input_data);
    num = floor(len/1000)*1000;
    n = (num/500)-3;
    for i=1:n
        if i==1
            sub_PPG(i,:) = input_data(i:i*2000);
         % 寻找所有大于阈值的波峰
            threshold=0.6*max(sub_PPG(i,:));
            [pks,locs] = findpeaks(sub_PPG(i,:),'minpeakheight',threshold);
        % 方差
            PPG_var(i,:)=var(pks);  
            SNR(i,:) = snr(sub_PPG(i,:),200);
        else 
            sub_PPG(i,:) = input_data((i-1)*500+1:(i+3)*500);
            % 寻找所有大于阈值的波峰
            threshold=0.6*max(sub_PPG(i,:));
            [pks,locs] = findpeaks(sub_PPG(i,:),'minpeakheight',threshold);
        % 方差
            PPG_var(i,:)=var(pks);
            SNR(i,:) = snr(sub_PPG(i,:),200);
        end  
        i = i+1;
    end

%最小值及下标
a = PPG_var;
row = find(a==0);
a(find(a==0))=[];
m=zeros(1,3);%索引值即下标
a_min=zeros(1,3);%最小值

m(1)=find(a==min(a));%找到第一个最小值得到它的索引值
a_min(1)=a(m(1));%找到第一个最小值

a(m(1))=max(a);%设定为一定比所有元素大的一个值，作为排除方式
m(2)=find(a==min(a));%找到第二个最小值得到它的索引值
a_min(2)=a(m(2));%找到第二个最小值

a(m(2))=max(a);%设定为一定比所有元素大的一个值，作为排除方式
m(3)=find(a==min(a));%找到第三个最小值得到它的索引值
a_min(3)=a(m(3));%找到第三个最小值
a(m(3))=max(a);%设定为一定比所有元素大的一个值，作为排除方式

PPG_good(1,:)=sub_PPG(m(1),:);
PPG_good(2,:)=sub_PPG(m(2),:);
PPG_good(3,:)=sub_PPG(m(3),:);
    
    
%最优信噪比最大值及下标
b = SNR;
b(find(b==0))=[];
S_dex=zeros(1,3);%索引值即下标
S_max=zeros(1,3);%最da值

S_dex(1)=find(b==max(b));%找到第一个最da值得到它的索引值
S_max(1)=b(S_dex(1));%找到第一个最da值

b(S_dex(1))=min(b);%设定为一定比所有元素大的一个值，作为排除方式
S_dex(2)=find(b==max(b));%找到第二个最小值得到它的索引值
S_max(2)=b(S_dex(2));%找到第二个最小值

b(S_dex(2))=min(b);%设定为一定比所有元素大的一个值，作为排除方式
S_dex(3)=find(b==max(b));%找到第三个最小值得到它的索引值
S_max(3)=b(S_dex(3));%找到第三个最小值
b(S_dex(3))=min(b);%设定为一定比所有元素大的一个值，作为排除方式

PPG_Sgood(1,:)=sub_PPG(S_dex(1),:);
PPG_Sgood(2,:)=sub_PPG(S_dex(2),:);
PPG_Sgood(3,:)=sub_PPG(S_dex(3),:);
end 