
load('SourceData.mat')
YSource=YSource*3-2;
XSource=dlarray(XSource,'CB');

InputAESize=size(XSource,1);
LayerNeckSize=24;
MyAElayer=[featureInputLayer(InputAESize)
    fullyConnectedLayer(496)
    fullyConnectedLayer(72)
    fullyConnectedLayer(LayerNeckSize,'Name','fc_Neck')
    fullyConnectedLayer(72)
    fullyConnectedLayer(496)
    fullyConnectedLayer(InputAESize)
    ];

FeatureSize=14;


MySOHlayer=[
    featureInputLayer(LayerNeckSize+FeatureSize)
    lstmLayer(128,'OutputMode','sequence')
    fullyConnectedLayer(64)
    fullyConnectedLayer(8)
    fullyConnectedLayer(1)
    ];


MyOptions = struct('MaxEpochs', 300, ...
    'InitialLearnRate', 1e-3, ...
    'ExecuEnvironment', 'gpu', ...% 使用GPU加速
     'L2Regularization', 0, ... % 加入L2正则化
     'ReplaceRate',0, ... 
     'Consheld', 0, ... % 贡献度激活阈值
     'Plots', 'training-progress'); % training-progress   none

SOHSAE = trainCustomNetwork(XSource,XSource,MyAElayer,MyOptions);
encoderNetSAE = dlnetwork(SOHSAE.Layers(1:4));
XSource_feature=forward(encoderNetSAE,XSource);
XSource_feature_Fused=[XSource_feature;zeros(FeatureSize,size(XSource,2))];


MyOptions.L2Regularization=0.005;
MyOptions.MaxEpochs=600;
MyOptions.ReplaceRate=2e-4;
MyOptions.Consheld=0.03;
SOHnet = trainCustomNetwork(XSource_feature_Fused(:,1:10:end),YSource(:,1:10:end),MySOHlayer,MyOptions);

YSource=YSource/3+2/3;

Ypred=forward(SOHnet,XSource_feature_Fused)/3+2/3;

subplot(1,2,1);
plot(YSource,'k');hold on;
plot(Ypred);
subplot(1,2,2);
plot(YSource,Ypred,'.');
rmse=sqrt(mean((YSource-Ypred).^2));
title(['RMSE= ',num2str(rmse*100),' (%)']);
