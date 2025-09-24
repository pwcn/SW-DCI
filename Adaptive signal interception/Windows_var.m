function [m,a_min,PPG_good,S_dex,S_max,PPG_Sgood] = Windows_var(input_data)
    len = length(input_data);
    num = floor(len/1000)*1000;
    n = (num/500)-3;
    for i=1:n
        if i==1
            sub_PPG(i,:) = input_data(i:i*2000);
         % Ѱ�����д�����ֵ�Ĳ���
            threshold=0.6*max(sub_PPG(i,:));
            [pks,locs] = findpeaks(sub_PPG(i,:),'minpeakheight',threshold);
        % ����
            PPG_var(i,:)=var(pks);  
            SNR(i,:) = snr(sub_PPG(i,:),200);
        else 
            sub_PPG(i,:) = input_data((i-1)*500+1:(i+3)*500);
            % Ѱ�����д�����ֵ�Ĳ���
            threshold=0.6*max(sub_PPG(i,:));
            [pks,locs] = findpeaks(sub_PPG(i,:),'minpeakheight',threshold);
        % ����
            PPG_var(i,:)=var(pks);
            SNR(i,:) = snr(sub_PPG(i,:),200);
        end  
        i = i+1;
    end

%��Сֵ���±�
a = PPG_var;
row = find(a==0);
a(find(a==0))=[];
m=zeros(1,3);%����ֵ���±�
a_min=zeros(1,3);%��Сֵ

m(1)=find(a==min(a));%�ҵ���һ����Сֵ�õ���������ֵ
a_min(1)=a(m(1));%�ҵ���һ����Сֵ

a(m(1))=max(a);%�趨Ϊһ��������Ԫ�ش��һ��ֵ����Ϊ�ų���ʽ
m(2)=find(a==min(a));%�ҵ��ڶ�����Сֵ�õ���������ֵ
a_min(2)=a(m(2));%�ҵ��ڶ�����Сֵ

a(m(2))=max(a);%�趨Ϊһ��������Ԫ�ش��һ��ֵ����Ϊ�ų���ʽ
m(3)=find(a==min(a));%�ҵ���������Сֵ�õ���������ֵ
a_min(3)=a(m(3));%�ҵ���������Сֵ
a(m(3))=max(a);%�趨Ϊһ��������Ԫ�ش��һ��ֵ����Ϊ�ų���ʽ

PPG_good(1,:)=sub_PPG(m(1),:);
PPG_good(2,:)=sub_PPG(m(2),:);
PPG_good(3,:)=sub_PPG(m(3),:);
    
    
%������������ֵ���±�
b = SNR;
b(find(b==0))=[];
S_dex=zeros(1,3);%����ֵ���±�
S_max=zeros(1,3);%��daֵ

S_dex(1)=find(b==max(b));%�ҵ���һ����daֵ�õ���������ֵ
S_max(1)=b(S_dex(1));%�ҵ���һ����daֵ

b(S_dex(1))=min(b);%�趨Ϊһ��������Ԫ�ش��һ��ֵ����Ϊ�ų���ʽ
S_dex(2)=find(b==max(b));%�ҵ��ڶ�����Сֵ�õ���������ֵ
S_max(2)=b(S_dex(2));%�ҵ��ڶ�����Сֵ

b(S_dex(2))=min(b);%�趨Ϊһ��������Ԫ�ش��һ��ֵ����Ϊ�ų���ʽ
S_dex(3)=find(b==max(b));%�ҵ���������Сֵ�õ���������ֵ
S_max(3)=b(S_dex(3));%�ҵ���������Сֵ
b(S_dex(3))=min(b);%�趨Ϊһ��������Ԫ�ش��һ��ֵ����Ϊ�ų���ʽ

PPG_Sgood(1,:)=sub_PPG(S_dex(1),:);
PPG_Sgood(2,:)=sub_PPG(S_dex(2),:);
PPG_Sgood(3,:)=sub_PPG(S_dex(3),:);
end 