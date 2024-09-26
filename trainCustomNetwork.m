function [net, info] = trainCustomNetwork(dlnet, XTrain, YTrain, options)
% V2.0 加入了成熟度的更新，L2正则化
    % 将数据移到 GPU 上
    dlX = gpuArray(XTrain);
    dlY = gpuArray(YTrain);
    
    % 训练网络
    numEpochs = options.MaxEpochs;
    InitialLearnRate=options.InitialLearnRate;
    L2Lamda=options.L2Regularization;
    if L2Lamda>0
        preWeights=dlnet.Learnables.Value;
    else
        preWeights=[];
    end
    % 初始化 Adam 优化器的动量变量
    % Adam 优化器的参数
    decayRate=0.001;
    
    updateRate=1e-6;
    % beta1 = 0.9;
    % beta2 = 0.999;
    % epsilon = 1e-8;
    %% 贡献度参数
    % 计算输出层 名字
    ThisConnections=dlnet.Connections(1:end-2,:);
    LayerIndex=find(strncmp(ThisConnections.Destination, 'fc_',3));%查找fc layer
    Layername=ThisConnections.Destination(LayerIndex);
    %计算权重层 索引
    weightIdx = find(strncmp(dlnet.Learnables.Layer, 'fc_',3) &...
                                strncmp(dlnet.Learnables.Parameter, 'Weights',7));
    weightIdx= weightIdx(1:end-1);
    weightIdx=dlarray(weightIdx,'CB');
    weightIdx=gpuArray(weightIdx);
    NodeMaturity=cell(1,length(weightIdx)); % 记录成熟度，迭代的次数
    NodeConSum = cell(1, length(weightIdx));
    NodeNum=0; % 神经元总个数

    for cnt=1:length(weightIdx)
        Temp=dlnet.Learnables.Value{weightIdx(cnt)};
        TempSize=size(Temp,1);
        NodeNum=NodeNum+TempSize;
        NodeConSum{cnt}=gpuArray(dlarray(0.5*ones(TempSize,1)));
        NodeMaturity{cnt}=gpuArray(dlarray(zeros(TempSize,1)));
    end
    SampleNum=dlarray(length(dlY)*NodeNum); % 每次更新的神经元总次数
    NodeTotalUpdateNum=0;
    % 记录损失值
    lossArray = zeros(1, numEpochs);
    NeuRate = zeros(1, numEpochs);
    % 创建训练进度监视器
    if strcmp(options.Plots, 'training-progress')
        show_flag = 1;
        monitor = trainingProgressMonitor;
        monitor.Metrics = ["TotalLoss", "Maturate"];
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
        % 前向传播和损失计算，使用 dlfeval
        [gradients, loss, loss_mse] = dlfeval(@modelGradients, dlnet, dlX, dlY,preWeights,L2Lamda);
        % 计算贡献度
        [NodeCon] = dlfeval(@ModelNodeContribution, dlnet, dlX,Layername,weightIdx);
        
        % 更新贡献度和成熟度
        for cnt = 1:length(NodeCon)
            NodeConSum{cnt} = 0.8*NodeConSum{cnt} + 0.2*NodeCon{cnt};
            NodeMaturity{cnt} = NodeMaturity{cnt}+1;
        end

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
            adamupdate( dlnet, gradients, averageGrad, averageSqGrad,epoch,learningRate);
        % 记录损失值
        lossArray(epoch) = extractdata(loss);
        % 检查更新节点
        NodeTotalUpdateNum=NodeTotalUpdateNum+SampleNum;
        if NodeTotalUpdateNum*updateRate>=1% 触发更新机制
           NodeTotalUpdateNum=0;
           nodeset=[];
           for cnt = 1:length(weightIdx)
               % 贡献度低于0.04 并且成熟度大于100
               XLogi=(NodeConSum{cnt} < 0.45 &NodeMaturity{cnt} >150);
               XLogi=find(extractdata(XLogi));
               if ~isempty(XLogi)
                   nodeset=[nodeset;[cnt*ones(length(XLogi),1) XLogi]];
               end
           end
           if ~isempty(nodeset)
               UpdateNodeindex=randi(size(nodeset,1));
               % 获取要更新的层数和节点索引
               selLayer = nodeset(UpdateNodeindex, 1);
               selNeuron = nodeset(UpdateNodeindex, 2);
               Layerindx=weightIdx(selLayer);
               % weight update
               ThisNeuronPara=dlnet.Learnables.Value{Layerindx}(selNeuron,:);
               ThisNeuronPara=randn(size(ThisNeuronPara), 'like', ThisNeuronPara)/3;
               dlnet.Learnables.Value{Layerindx}(selNeuron,:)=ThisNeuronPara;
               % bias update
               ThisNeuronPara=dlnet.Learnables.Value{Layerindx+1}(selNeuron,:);
               ThisNeuronPara=randn([1 1], 'like', ThisNeuronPara)/3;
               dlnet.Learnables.Value{Layerindx+1}(selNeuron,:)=ThisNeuronPara;
               NodeMaturity{selLayer}(selNeuron)=0;
           end
        end
        TRate=0;
       for cnt = 1:length(weightIdx)
               % 贡献度低于0.04 并且成熟度大于100
               TRate=TRate + sum((NodeConSum{cnt} > 0.5));
       end
       NeuRate(epoch)=TRate/NodeNum;
        if show_flag
            recordMetrics(monitor, epoch, "TotalLoss", loss, "Maturate", TRate/NodeNum);
            monitor.Progress = 100 * epoch / numEpochs;
        end
    end

    % 返回训练后的网络和损失信息
    net = dlnet;
    info = struct('Loss', lossArray);
    info.ManuRate=NeuRate;
end


function [gradients, loss, loss_mse] = modelGradients(dlnet, dlX, dlY,preWeights,L2Lamda)
    % 前向传播
     dlYpred = forward(dlnet, dlX);
    % 计算损失
    Error=dlYpred - dlY;
    loss_mse = mean((Error).^2);
    L2Loss=0;
    if L2Lamda<=0
    for cnt=1:length(preWeights)
        L2Loss=L2Loss+sum(sum((dlnet.Learnables.Value{cnt}-preWeights{cnt}).^2));
    end
    end
    loss=mean((Error).^2)+ L2Lamda*L2Loss;%tanh(loss_mse);
    % 计算梯度
    gradients = dlgradient(loss, dlnet.Learnables);
end

function [NodeCon] = ModelNodeContribution(dlnet, dlX,Layername,Windx)
% 计算节点权重 NodeCon
LayerCnt=length(Layername);
HiddenOut=cell(1,LayerCnt);
LocalWeight=cell(1,LayerCnt);
NodeCon=cell(1,LayerCnt);
for cnt=1:LayerCnt
    HiddenOut{cnt}= forward(dlnet, dlX,'Outputs',Layername{cnt});
    LocalWeight{cnt}=dlnet.Learnables.Value{Windx(cnt)};
    colSums = sum(abs(LocalWeight{cnt}), 2);
    NodeCon{cnt}=mean(colSums.*abs(HiddenOut{cnt}),2);
end
end
