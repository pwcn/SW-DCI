% ���ܣ���һά�źŵĸ�˹�˲���ͷβr/2���źŲ������˲�
% r     :��˹ģ��Ĵ�С�Ƽ�����
% sigma :��׼��
% y     :��Ҫ���и�˹�˲�������
function y_filted = Gaussianfilter(input)

% ����һά��˹�˲�ģ��
r        = 3;
sigma    = 1;

GaussTemp = ones(1,r*2-1);
for i=1 : r*2-1
    GaussTemp(i) = exp(-(i-r)^2/(2*sigma^2))/(sigma*sqrt(2*pi));
end

% ��˹�˲�
y_filted = input;
for i = r : length(input)-r+1
    y_filted(i) = input(i-r+1 : i+r-1)*GaussTemp';
end
