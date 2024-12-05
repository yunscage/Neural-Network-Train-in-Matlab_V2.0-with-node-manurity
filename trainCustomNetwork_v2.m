function [net, info] = trainCustomNetwork_v2(XTrain, YTrain, dlLayers,options)
dlnet=dlnetwork(dlLayers);
% 将数据移到 GPU 上
if options.ExecuEnvironment=='gpu'
    dlX = gpuArray(XTrain);
    dlY = gpuArray(YTrain);
end
% 训练网络
numEpochs = options.MaxEpochs;
InitialLearnRate=options.InitialLearnRate;
L2Params.L2Lamda=options.L2Regularization;
L2Params.LearnWeights=dlnet.Learnables.Value;
if L2Params.L2Lamda<=0
    L2Params.LearnWeights=[];
end
updateRate=options.updateRate;
ConSheld=options.Consheld;% 贡献度更新百分比
% 初始化 Adam 优化器的动量变量
% Adam 优化器的参数
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
    
    NodeNum=0; % 神经元总个数
    % 成熟度记录
    NodeComSum=0;% 更新神经元个数
    NodeConSum = ModelNodeContribution(dlnet, dlX(:,1),Layername,weightIdx);
    NodeMaturity=cell(1,LayerCnt); % 记录成熟度，迭代的次数
    for cnt=1:LayerCnt
        NodeMaturity{cnt} = zeros(size(NodeConSum{cnt}));
        NodeNum=NodeNum + length(NodeConSum{cnt});
    end
    SampleNum=dlarray(NodeNum); % 每次更新的神经元总次数
    NodeTotalUpdateNum=0;
% 记录损失值
lossArray = zeros(1, numEpochs);

% 创建训练进度监视器
if strcmp(options.Plots, 'training-progress')
    show_flag = 1;
    monitor = trainingProgressMonitor;
    monitor.Metrics = ["TotalLoss","Manurity","NodeUpdate"];
    monitor.XLabel = "Epoch";
else
    show_flag = 0;
end
averageGrad=[];
averageSqGrad=[];
% 开始训练循环
for epoch = 1:numEpochs
    % 动态调整学习率
    learningRate = InitialLearnRate * (1 / (1 + decayRate * epoch));
    updateRate = updateRate*0.995;
    % 前向传播和损失计算，使用 dlfeval
    [gradients, loss] = dlfeval(@modelGradients, dlnet, dlX, dlY,L2Params);

    % 遍历 gradients 表中的每个元素，并添加噪声
    for cnt = 1:size(gradients, 1)
        % 提取当前的梯度值
        gradValue = gradients.Value{cnt};
        % 生成与当前梯度维度匹配的随机噪声，并确保其类型与梯度一致  GPU or CPU
        noise = 0.2*learningRate * randn(size(gradValue), 'like', gradValue);  % 'like' 保证类型一致
        % 将噪声添加到梯度值中
        gradients.Value{cnt} = gradValue + noise;
    end

    % Adam Update / Adam 更新
    [dlnet, averageGrad, averageSqGrad] = ...
        adamupdate( dlnet, gradients, averageGrad, averageSqGrad,epoch,learningRate,...
        beta1,beta2,epsilon);

    %% 贡献度相关计算
    % 计算贡献度
    NodeCon = ModelNodeContribution(dlnet, dlX,Layername,weightIdx);

    % 更新贡献度和成熟度
    for cnt = 1:LayerCnt
        NodeConSum{cnt} = 0.7*NodeConSum{cnt} + 0.3*NodeCon{cnt};
        NodeMaturity{cnt} = NodeMaturity{cnt}+1;
    end
    % 检查更新节点
    NodeTotalUpdateNum=NodeTotalUpdateNum+SampleNum;
    if NodeTotalUpdateNum*updateRate>=1% 触发更新机制
        NodeTotalUpdateNum=0;
        nodeset=[];
        for cnt = 1:LayerCnt
            % 检查节点贡献度
            XLogi=(NodeConSum{cnt} < ConSheld &NodeMaturity{cnt} >50);
            XLogi=find(extractdata(XLogi));
            if ~isempty(XLogi)
                nodeset=[nodeset;[cnt*ones(length(XLogi),1) XLogi]];
            end
        end
        if ~isempty(nodeset)
            UpdateNodeindex=randi(size(nodeset,1));
            % 获取要更新的层数和节点索引
            selLayer = nodeset(UpdateNodeindex, 1);% 层数
            selNeuron = nodeset(UpdateNodeindex, 2);% 节点索引
            dlnet = ResetNeuronParameters(dlnet, Layername(selLayer), selNeuron);
            NodeComSum=NodeComSum+1;
            NodeMaturity{selLayer}(selNeuron)=0;
        end
    end
    TRate=0;
    for cnt = 1:LayerCnt
        % 贡献度低于0.04 并且成熟度大于100
        TRate=TRate + sum((NodeConSum{cnt} > 1.1*ConSheld));
    end


    lossArray(epoch) = extractdata(loss);% 记录损失值
    if show_flag
        recordMetrics(monitor, epoch, "TotalLoss", loss,...
            "Manurity",TRate/NodeNum,...
            "NodeUpdate",NodeComSum);
        monitor.Progress = 100 * epoch / numEpochs;
    end

end

% 返回训练后的网络和损失信息


net = dlnet;
info = struct('Loss', lossArray);
end


function [gradients, loss] = modelGradients(dlnet, dlX, dlY,L2Params)
% 前向传播
dlYpred = forward(dlnet, dlX);
% 计算损失
Error=dlYpred - dlY;
loss_mse = mean((Error).^2);
L2Loss=0;
L2Lamda=L2Params.L2Lamda;
if L2Lamda<=0
    preWeights=L2Params.LearnWeights;
    for cnt=1:length(preWeights)
        L2Loss=L2Loss+sum(sum((dlnet.Learnables.Value{cnt}-preWeights{cnt}).^2));
    end
end
loss=loss_mse+ L2Lamda*L2Loss;
% 计算梯度
gradients = dlgradient(loss, dlnet.Learnables);
end

function [NodeCon] = ModelNodeContribution(dlnet, dlX,Layername,Windx)
% 计算节点权重 NodeCon
LayerCnt=length(Layername);
NodeCon=cell(1,LayerCnt);
for cnt=1:LayerCnt
    HiddenOut= forward(dlnet, dlX,'Outputs',Layername{cnt});% Output hidden stste
    HidOut=mean(abs(stripdims(HiddenOut)),2);
    LocalWeight=dlnet.Learnables.Value{Windx(cnt)};
    colSums = sum(abs(LocalWeight), 1)';
    NodeCon{cnt}=colSums.*HidOut;
end
end

function dlnet = ResetNeuronParameters(dlnet, layerName, neuronIndex)
% 更新 dlnet 中指定层的特定神经元的参数
%
% 输入参数:
% dlnet          - 深度学习网络对象 (dlnetwork)
% layerName      - 要更新的层的名称 (字符串)
% neuronIndex    - 要更新的神经元索引 (整数)
%
% 输出参数:
% dlnet          - 更新后的深度学习网络对象

% 查找目标层参数
LayerParams = dlnet.Learnables;
rows = LayerParams.Layer == layerName;
Para_Value=LayerParams.Value;
% 更新 FullyConnected 层的参数
Sel_Row = LayerParams.Parameter == "Weights";
if any(rows & Sel_Row)
    % 获取并更新 Weights
    Sel_Para = Para_Value{rows & Sel_Row};
    Sel_Para(neuronIndex, :) = randn(size(Sel_Para(neuronIndex, :)), 'like', Sel_Para)/3;
    Para_Value{rows & Sel_Row}= Sel_Para;

    % 获取并更新 Bias
    Sel_Row = LayerParams.Parameter == "Bias";
    Sel_Para = Para_Value{rows & Sel_Row};
    Sel_Para(neuronIndex, :) = randn(size(Sel_Para(neuronIndex, :)), 'like', Sel_Para)/3;% 更新偏置
    Para_Value{rows & Sel_Row} = Sel_Para;
end

% 更新 LSTM 层的参数
Sel_Row = LayerParams.Parameter == "InputWeights";
if any(rows & Sel_Row)
    % 获取并更新 InputWeights
    lstmneuronIndex=4*(neuronIndex-1)+1:4*neuronIndex;
    Sel_Para = Para_Value{rows & Sel_Row};
    Sel_Para(lstmneuronIndex, :) = randn(size(Sel_Para(lstmneuronIndex, :)), 'like', Sel_Para)/3;
    Para_Value{rows & Sel_Row} = Sel_Para;

    % 获取并更新 RecurrentWeights
    Sel_Row = LayerParams.Parameter == "RecurrentWeights";
    Sel_Para = Para_Value{rows & Sel_Row};
    Sel_Para(lstmneuronIndex, :) = randn(size(Sel_Para(lstmneuronIndex, :)), 'like', Sel_Para)/3;
    Para_Value{rows & Sel_Row} = Sel_Para;

    % 获取并更新 Bias
    Sel_Row = LayerParams.Parameter == "Bias";
    Sel_Para = Para_Value{rows & Sel_Row};
    Sel_Para(lstmneuronIndex, :) = randn(size(Sel_Para(lstmneuronIndex, :)), 'like', Sel_Para)/3; % 更新偏置
    Para_Value{rows & Sel_Row} = Sel_Para;
end
dlnet.Learnables.Value=Para_Value;
end
