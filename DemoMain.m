load("Demo3Data.mat");
Mylayers=[
    featureInputLayer(14)
    % lstmLayer(14,"OutputMode","sequence");
    fullyConnectedLayer(108)
    fullyConnectedLayer(64)
    fullyConnectedLayer(32)
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(1)
    % regressionLayer
    ];

MyOptions = struct('MaxEpochs', 600, ...
    'InitialLearnRate', 1e-3, ...
    'ExecuEnvironment', 'gpu', ...% 使用GPU加速
     'L2Regularization', 0, ... % 加入L2正则化
     'updateRate',2.5e-5, ... 
     'Consheld', 0.05, ... % 贡献度激活阈值
     'Plots', 'training-progress'); % training-progress none

ThisDataX=[];
ThisDataY=[];
PreDataX=DataX{1};
PreDataY=DataY{1}/DataY{1}(1);
PreXTrain = dlarray(PreDataX, 'CB');
PreYTrain = dlarray(PreDataY, 'CB');
for cnt=1:length(DataY)
    ThisDataX=DataX{cnt};
    ThisDataY=DataY{cnt}/DataY{cnt}(1);


    %% 使用自定义函数更新神经网络
    Prenet=dlnetwork(Mylayers);
    XTrain = dlarray(ThisDataX, 'CB');
    YTrain = dlarray(ThisDataY, 'CB');
    if cnt==1
        Thisnet=trainCustomNetwork_v2(XTrain,YTrain,Mylayers,MyOptions);
    else
        MyOptions.InitialLearnRate=5e-4;
        MyOptions.L2Regularization=5e-4;
        Thisnet=trainCustomNetwork_v2(XTrain,YTrain,Thisnet.Layers,MyOptions);
    end
    ypred=forward(Thisnet,XTrain);

    ypredpre=forward(Thisnet,PreXTrain);
    figure;
    plot(ThisDataX(7,:),ThisDataY,'k');hold on;
    plot(ThisDataX(7,:),ypred,'r');

    rmsetotal=extractdata(sqrt(mean((ThisDataY-ypred).^2)));
    rmse1=extractdata(sqrt(mean((PreYTrain-ypredpre).^2)));
    disp('******Result******');
    disp('RMSEtotal=');
    disp(rmsetotal);
    disp('RMSE1=');
    disp(rmse1);
    pause(0.5);
end
