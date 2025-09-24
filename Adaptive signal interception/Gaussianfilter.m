% 功能：对一维信号的高斯滤波，头尾r/2的信号不进行滤波
% r     :高斯模板的大小推荐奇数
% sigma :标准差
% y     :需要进行高斯滤波的序列
function y_filted = Gaussianfilter(input)

% 生成一维高斯滤波模板
r        = 3;
sigma    = 1;

GaussTemp = ones(1,r*2-1);
for i=1 : r*2-1
    GaussTemp(i) = exp(-(i-r)^2/(2*sigma^2))/(sigma*sqrt(2*pi));
end

% 高斯滤波
y_filted = input;
for i = r : length(input)-r+1
    y_filted(i) = input(i-r+1 : i+r-1)*GaussTemp';
end
