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

MyOptions = trainingOptions('adam', ...
    'MaxEpochs', 600, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 1e-3, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 0, ... % 加入L2正则化
    'ExecutionEnvironment', 'gpu'); % 使用GPU加速

ThisDataX=[];
ThisDataY=[];
PreDataX=DataX{1};
PreDataY=DataY{1}/DataY{1}(1);
PreXTrain = dlarray(PreDataX, 'CB');
PreYTrain = dlarray(PreDataY, 'CB');
for cnt=1:length(DataY)
    ThisDataX=DataX{cnt};
    ThisDataY=DataY{cnt}/DataY{cnt}(1);

     % plot(DataX{cnt}(7,:),DataY{cnt}/DataY{cnt}(1));hold on;
     % title(num2str(cnt));

% ThisDataX(7,:)=ThisDataX(7,:)/2;
% ThisDataX(8,:)=ThisDataX(8,:)/3;
% ThisDataX(9,:)=ThisDataX(9,:)/3;
%% 使用自定义函数更新神经网络
Prenet=dlnetwork(Mylayers);
XTrain = dlarray(ThisDataX, 'CB'); 
YTrain = dlarray(ThisDataY, 'CB'); 
if cnt==1
    Thisnet=trainCustomNetwork(Prenet,XTrain,YTrain,MyOptions);
else
    MyOptions.InitialLearnRate=5e-4;
    MyOptions.L2Regularization=5e-4;
    Thisnet=trainCustomNetwork(Thisnet,XTrain,YTrain,MyOptions);
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
