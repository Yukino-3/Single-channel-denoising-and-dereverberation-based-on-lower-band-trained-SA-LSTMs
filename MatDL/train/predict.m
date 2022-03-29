function [yplabel] = predict(x, fun, model, opt)
%PREDICT Evaluate predictions of a model
%   Inputs:
%       - x: input to model, of size: batch size (m) x input dimensions (d)
%       - fun: a function handle to the model
%       - model: a structure of model weights
%       - opt: a structure of hyper-parameters
%   Outputs:
%       - yplabel: predicted class label, of size: batch size (m) x 1
%       - confidence: confidence of predicted class label, of size: batch size (m) x 1
%       - classes: sorted class predictions, of size: batch size (m) x number of classes (K)
%       - classConfidences: confidence of sorted class predictions, of size: batch size (m) x number of classes (K)
%       - yp: class predictions, of size: batch size (m) x number of classes (K)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.
    
    
        yplabel = fun(x, 0,model,  0,opt);
   
%     [confidence, yplabel] = max(yp,[],2);
%     [classConfidences, classes] = sort(yp, 2, 'descend');

end
