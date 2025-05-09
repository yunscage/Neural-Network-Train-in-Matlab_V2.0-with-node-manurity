function [net, info] = trainCustomNetwork(dlX, dlY,dlLayers,options)
dlnet=dlnetwork(dlLayers);
if options.ExecuEnvironment=='gpu'
    dlX=gpuArray(dlX);
    dlY=gpuArray(dlY);
end
% 训练网络
numEpochs = options.MaxEpochs;
InitialLearnRate=options.InitialLearnRate;
L2Params.L2Lamda=options.L2Regularization;
L2Params.LearnWeights=dlnet.Learnables.Value;

if L2Params.L2Lamda<=0
    L2Params.LearnWeights=[];
end

% 初始化 Adam 优化器的动量变量
decayRate = 0.001;  % 学习率衰减
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
    
%计算权重层 索引
LayerCntIdx = find(strncmp(dlnet.Learnables.Parameter, 'Weights',7) |...
    strncmp(dlnet.Learnables.Parameter, 'InputWeights',12));
LayerCnt=length(LayerCntIdx)-1;
% 计算输出层 名字
Layername= unique(dlnet.Learnables.Layer, 'stable'); % 顺序提取不重复的层名
Layername=Layername(1:end-1);
weightIdx= LayerCntIdx(2:end);
weightIdx=gpuArray(weightIdx);

LayerNodeNum=0; % 神经元总个数
% 成熟度记录
NodeComSum=0;% 更新神经元个数
NodeConSum = ModelNodeContribution(dlnet, dlX(:,1),Layername,weightIdx);
NodeAge=cell(1,LayerCnt); % 记录每一层成熟度，迭代的次数
CLayerNum2replace=zeros(1,LayerCnt); % 每层需要更新节点数量 number of units to replace in layer

for cnt=1:LayerCnt
    NodeAge{cnt} = zeros(size(NodeConSum{cnt}));
    LayerNodeNum=LayerNodeNum + length(NodeConSum{cnt});
end

InitialReplaceRate=options.ReplaceRate;

AgeDeadline=80;% 节点的年龄成熟阈值
CBPPara.NodeConSum=NodeConSum;
CBPPara.NodeComSum=NodeComSum;
CBPPara.NodeAge=NodeAge;
InitialPercentSheld=options.Consheld;% 贡献度阈值的百分比
CBPPara.PercentSheld=InitialPercentSheld;

% 记录损失值
lossArray = zeros(1, numEpochs);
NeuRate = zeros(1, numEpochs);
Manurity = zeros(1, numEpochs);
% 创建训练进度监视器
if strcmp(options.Plots, 'training-progress')
    show_flag = 1;
    monitor = trainingProgressMonitor;
    monitor.Metrics = ["TotalLoss", "Manurity", "NodeUpdate"];
    monitor.XLabel = "Epoch";
else
    show_flag = 0;
end
averageGrad=[];
averageSqGrad=[];
% 开始训练循环
for epoch = 1:numEpochs
    % 动态调整学习率
    DecrayGain=(1 + decayRate * epoch);
    learningRate = InitialLearnRate/DecrayGain;
    CBPPara.PercentSheld = InitialPercentSheld /DecrayGain;
    ReplaceRate = InitialReplaceRate/DecrayGain;
    % 前向传播和损失计算，使用 dlfeval
    [gradients, loss] = dlfeval(@modelGradients,dlnet, dlX, dlY,L2Params);

    % 遍历 gradients 表中的每个元素，并添加噪声
    for cnt = 1:size(gradients, 1)
        % 提取当前的梯度值
        gradValue = gradients.Value{cnt};
        % 生成与当前梯度维度匹配的随机噪声，并确保其类型与梯度一致  GPU or CPU
        noise = 0.5*learningRate * randn(size(gradValue), 'like', gradValue);  % 'like' 保证类型一致
        % 将噪声添加到梯度值中
        gradients.Value{cnt} = gradValue + noise;
    end

    % Adam Update / Adam 更新
    [dlnet, averageGrad, averageSqGrad] = ...
        adamupdate( dlnet, gradients, averageGrad, averageSqGrad,epoch,learningRate,...
        beta1,beta2,epsilon);

    %% 贡献度相关计算
    % 计算贡献度
    [NodeCon] = ModelNodeContribution(dlnet, dlX,Layername,weightIdx);

   % 更新贡献度和成熟度
    TRate=0;
    NodeAge=CBPPara.NodeAge;
    NodeConSum=CBPPara.NodeConSum;
    for cnt = 1:LayerCnt
        NodeConSum{cnt} = 0.99*NodeConSum{cnt} + 0.01*NodeCon{cnt};
        NodeAge{cnt} = NodeAge{cnt}+1;
        CLayerNum2replace(cnt)=CLayerNum2replace(cnt)+sum(NodeAge{cnt}>AgeDeadline);% 
    end

    NodeComSum=CBPPara.NodeComSum;
    CBPPara.NodeConSum=NodeConSum;
    CBPPara.NodeAge=NodeAge;
    % NeuRate(epoch)=TRate/LayerNodeNum;
    % 检查更新节点
    ReSetIndexs=CLayerNum2replace*ReplaceRate>=1;

    if any(ReSetIndexs) && epoch<numEpochs-100% 触发更新机制
        [dlnet,CBPPara]=Update_CBP(dlnet,CBPPara,Layername,ReSetIndexs);
        CLayerNum2replace(ReSetIndexs)=0;
    end
    % 
    NetNodeConSum=0;
    for cnt=1:LayerCnt
        NetNodeConSum=NetNodeConSum+sum(NodeConSum{cnt});
    end
    NetNodeConSum=NetNodeConSum/LayerNodeNum;

    % 记录损失值
    lossArray(epoch) = extractdata(loss);
    Manurity(epoch) = extractdata(NetNodeConSum);
    if show_flag
        recordMetrics(monitor, epoch, "TotalLoss", loss, ...
            "Manurity", NetNodeConSum, ... % TRate/NodeNum
            "NodeUpdate", NodeComSum);  % NodeComSum
        monitor.Progress = 100 * epoch / numEpochs;
    end
end


% 返回训练后的网络和损失信息
net = dlnet;
info = struct('Loss', lossArray);
% info.ManuRate=NeuRate;
info.Manurity=Manurity;
info.NodeConSum=NodeConSum;
end


function [gradients, loss] = modelGradients(dlnet, dlX0, dlY0,L2Params)
% 前向传播
dlYpred0 = forward(dlnet, dlX0);
Loss_total0= Loss_fcn_cal(dlYpred0,dlY0);
L2Loss=0;
L2Lamda=L2Params.L2Lamda;
if L2Lamda<=0
    preWeights=L2Params.LearnWeights;
    for cnt=1:length(preWeights)
        weight_bias=(dlnet.Learnables.Value{cnt}-preWeights{cnt}).^2;
        L2Loss=L2Loss+mean(weight_bias(:));
    end
end

loss=Loss_total0  + L2Lamda*L2Loss ;%tanh(loss_mse);
% 计算梯度
gradients = dlgradient(loss, dlnet.Learnables);
end


function Loss_total= Loss_fcn_cal(dlY,dlYpred)
% 计算标签损失
Error=dlYpred - dlY;
Error=Error(:).^2;
loss_mse=mean(Error);
Loss_total = loss_mse;
end