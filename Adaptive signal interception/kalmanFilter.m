function z = kalmanFilter(x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  function z = kalmanFilter(x)
%
%>
%> @brief 一维卡尔曼滤波
%>
%> @param[out]  z             滤波后的结果
%> @param[in]   x             需要滤波的数据
%>
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    % 卡尔曼相关变量定义
    persistent xk xk_1;             % 状态量
    persistent zk;                  % 观测量
    persistent A;                   % 状态转移矩阵
%     persistent B;                   % 控制输入模型
    persistent H;                   % 观测矩阵
    persistent Pk Pk_1;             % 误差协方差矩阵
    persistent Q;                   % 状态噪声协方差矩阵
    persistent R;                   % 观测噪声协方差矩阵
    
    % 卡尔曼相关参数初始化
    if isempty(xk)
        A = 1;
        H = 1;
        Pk = 1;
        Pk_1 = 1;                   % 初始误差协方差为1
        Q = 0.01;                   % 反应两个时刻rssi方差
        R = 0.05;                   % 反应测量rssi的测量精度
        xk = 0;
        xk_1 = 0;
        zk = 0;
    end
    
    I = 1;
    if xk_1 == 0
        xk_1 = x;
        xk = x;
    else
        zk = H*x;                   % 观测量方程
        % 预测
        X = A*xk_1;                 % 状态预测
        P = A*Pk_1*A' + Q;          % 误差协方差预测
        % 更新(校正)
        K = P*H'*inv(H*P*H'+R);     % 卡尔曼增益更新
        xk = X + K*(zk - H*X);      % 更新校正
        xk_1 = xk;                  % 保存校正后的值，下一次滤波使用
        Pk = (I - K*H)*P;           % 更新误差协方差
        Pk_1 = Pk;                  % 保存校正后的误差协方差，下一次滤波使用
    end
    
    % 滤波结果返回
    z = xk;
end