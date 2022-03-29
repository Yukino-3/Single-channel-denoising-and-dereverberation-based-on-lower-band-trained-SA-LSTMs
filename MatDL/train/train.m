function [ model, opt ] = train( X, Y,M,model, lossfun, opt )
%TRAIN Train a model
%   Inputs:
%       - X: training dataset, of size: number of data points (M) x input dimensions (d)
%       - Y: training labels, of size: number of data points (M) x number of classes (K)
%       - XVal: validation dataset, of size: number of data points (M) x input dimensions (d)
%       - YVal: validations labels, of size: number of data points (M) x number of classes (K)
%       - model: a structure of model weights
%       - lossfun: a function handle to the model
%       - opt: a structure of hyper-parameters
%   Outputs:
%       - model: a structure of trained model weights
%       - trainLoss: training loss history
%       - trainAccuracy: training accuracy history
%       - valLoss: validation loss history
%       - valAccuracy: validation accuracy history
%       - opt: a structure of updated hyper-parameters
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

%% Initialize

        [loss, grads, opt] = lossfun(X, M,model, Y, opt); % Fprop & BackProp , the model is the LSTM
        optim = opt.optim;
        % Update model
        [model, opt] = optim(model, grads, opt);
end