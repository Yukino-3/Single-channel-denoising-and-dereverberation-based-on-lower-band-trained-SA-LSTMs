% A complete example of a recurrent neural network
%% Init
clear all
addpath(genpath('../MatDL'));

%% Load data
load('mnist_uint8.mat');
X = double(reshape(train_x',28,28,60000))/255; X = permute(X, [3 2 1]);  % 60000*28*28
XVal = double(reshape(test_x',28,28,10000))/255; XVal = permute(XVal, [3 2 1]);
Y = double(train_y);
YVal = double(test_y);

%% Initialize model
opt = struct;

[model, opt] = init_lstm_rnn(28, 10, [15 15 ], opt);  % (input size, class, size of hidden layer, opts)

%% Hyper-parameters
opt.batchSize = 100;  

opt.optim = @rmsprop;  % RMSPROP Do an RMS prop update step
% opt.beta1 = 0.9; opt.beta2 = 0.999; opt.t = 0; opt.mgrads = opt.vgrads;
opt.rmspropDecay = 0.99;
% opt.initialMomentum = 0.5; opt.switchEpochMomentum = 1; opt.finalMomentum = 0.9;
opt.learningRate = 0.001;  % learning rate
opt.learningDecaySchedule = 'exp'; % 'no_decay', 't/T', 'step'
opt.learningDecayRate = 1;
% opt.learningDecayRateStep = 5;

opt.dropout = 1;  % dropout or not
opt.weightDecay = false;
opt.maxNorm = false;

opt.maxEpochs = 3;   % number of epochs
opt.earlyStoppingPatience = 5;
opt.valFreq = 100;

opt.plotProgress = false;
opt.extractFeature = false;
opt.computeDX = false;

opt.useGPU = false;

%% Gradient check
% x = X(1:100, :, :);   % select part of x to validation
% y = Y(1:100, :, :);   % select part of x to validation
% maxRelError = gradcheck(@lstm_rnn, x, model, y, opt, 5);

%% Train
[model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt] = train( X, Y, XVal, YVal, model, @lstm_rnn, opt );

%% Predict
[yplabel] = predict(XVal, @lstm_rnn, model, opt);