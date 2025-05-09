function [dlnet,CBPPara]=Update_CBP(dlnet,CBPPara,Layername,ReSetIndexs)
% Udpate in the continual backpropagation
NodeAge=CBPPara.NodeAge;
NodeConSum=CBPPara.NodeConSum;
PercentSheld=CBPPara.PercentSheld;

PercentSheld=min(PercentSheld,0.2);
for selLayer = find(ReSetIndexs>0)
    % 检查节点贡献度
    ThisLayerNodeCon=extractdata(NodeConSum{selLayer});
    % 选择一定阈值的数据计算
    NumberForReset=ceil(PercentSheld*length(ThisLayerNodeCon));
    if NumberForReset<=0
        return;
    end
    [~,Sortindex]=sort(ThisLayerNodeCon);
    Sel_Unit=Sortindex(NumberForReset);
    ThisConSheld=ThisLayerNodeCon(Sel_Unit);
    selNeuron=find(ThisLayerNodeCon <= ThisConSheld & NodeAge{selLayer});    
    dlnet = ResetNeuronParameters(dlnet, Layername(selLayer), selNeuron);
    CBPPara.NodeComSum=CBPPara.NodeComSum+1;
    NodeAge{selLayer}(selNeuron)=0;
end

CBPPara.NodeAge=NodeAge;
end

function dlnet = ResetNeuronParameters(dlnet, layerName, neuronIndex)
% 更新 dlnet 中指定层的特定神经元的参数
%
% 输入参数:
% dlnet          - 深度学习网络对象 (dlnetwork)
% layerName      - 要更新的层的名称 (字符串)
% neuronIndex    - 要更新的神经元索引 (整数)
% Resetflag      - 随机更新或者清零
%
% 输出参数:
% dlnet          - 更新后的深度学习网络对象

% 查找目标层参数
LayerParams = dlnet.Learnables;
rows = LayerParams.Layer == layerName;
Para_Value=LayerParams.Value;
% 更新 LSTM 层的参数
Sel_Row = LayerParams.Parameter == "InputWeights";
if any(rows)
    if (rows&Sel_Row)
        % 针对LSTM神经网络
        neuronIndex=4*(neuronIndex-1)+1:4*neuronIndex;
    end
    Sel_Para = Para_Value(rows);
    for cnt=1:length(Sel_Para)
        Sel_Para{cnt}(neuronIndex, :) = randn(size(Sel_Para{cnt}(neuronIndex, :)),...
            'like',Sel_Para{cnt});
    end
    Para_Value(rows)=Sel_Para;
    % 控制输出连接的权重为随机数
    NextLayerCnt= find(diff(rows)<0,1)+1;
    % %
    Para_Value{NextLayerCnt}(:,neuronIndex)=0;
end
dlnet.Learnables.Value=Para_Value;
end
