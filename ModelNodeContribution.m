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