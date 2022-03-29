function [out, grads, opt] = five_convnet(X, model, y, opt)

    weightDecay = opt.weightDecay; 
    dropout = opt.dropout;
    extractFeature = opt.extractFeature;
    computeDX = opt.computeDX;

    w1 = model.w1; b1 = model.b1;
    w2 = model.w2; b2 = model.b2;
    w3 = model.w3; b3 = model.b3;
    w4 = model.w4; b4 = model.b4;
    w5 = model.w5; b5 = model.b5;
    
    gamma1 = model.gamma1; beta1 = model.beta1;
    gamma2 = model.gamma2; beta2 = model.beta2;
    gamma3 = model.gamma3; beta3 = model.beta3;
    gamma4 = model.gamma4; beta4 = model.beta4;
    
    bnParam1 = opt.bnParam1; bnParam2 = opt.bnParam2;
    bnParam3 = opt.bnParam3; bnParam4 = opt.bnParam4;
    
    convParam.stride = 1; convParam.pad = 0; convParam.useGPU = opt.useGPU;
    
    poolParam1.stride = 2; poolParam1.poolHeight = 2; poolParam1.poolWidth = 2;
    poolParam2.stride = 2; poolParam2.poolHeight = 2; poolParam2.poolWidth = 2;
    poolParam1.useGPU = opt.useGPU; poolParam2.useGPU = opt.useGPU;
    
    dropoutParam.p = dropout;
    if gather(y == 0), dropoutParam.mode = 'test'; else dropoutParam.mode = 'train'; end
    dropoutParam.useGPU = opt.useGPU;
    
    if gather(y == 0) 
        bnParam1.mode = 'test'; bnParam2.mode = 'test';
        bnParam3.mode = 'test'; bnParam4.mode = 'test';
    else
        bnParam1.mode = 'train'; bnParam2.mode = 'train';
        bnParam3.mode = 'train'; bnParam4.mode = 'train';
    end
    
    [a1, bnParam1, cache1] = conv_bn_relu_pool_forward(X, w1, b1, gamma1, beta1, bnParam1, convParam, poolParam1);
    [a2, bnParam2, cache2] = conv_bn_relu_pool_forward(a1, w2, b2, gamma2, beta2, bnParam2, convParam, poolParam2);
    [a3, bnParam3, cache3] = affine_bn_relu_forward(a2, w3, b3, gamma3, beta3, bnParam3);
    [d3, cached3] = dropout_forward(a3, dropoutParam);
    [a4, bnParam4, cache4] = affine_bn_relu_forward(d3, w4, b4, gamma4, beta4, bnParam4);
    [d4, cached4] = dropout_forward(a4, dropoutParam);
    [scores, cache5] = affine_forward(d4, w5, b5);
    
    if (extractFeature), out = scores; grads = 0; return; end
    
    yp = softmax_forward(scores);
    
    if gather(y == 0), out = yp; grads = 0; return; end
    
    [dataLoss, dscores] = softmax_backward(yp, y);
    [dd4, dw5, db5] = affine_backward(dscores, cache5);
    da4 = dropout_backward(dd4, cached4);
    [dd3, dw4, db4, dGamma4, dBeta4] = affine_bn_relu_backward(da4, cache4);
    da3 = dropout_backward(dd3, cached3);
    [da2, dw3, db3, dGamma3, dBeta3] = affine_bn_relu_backward(da3, cache3);
    [da1, dw2, db2, dGamma2, dBeta2] = conv_bn_relu_pool_backward(da2, cache2);
    [dX, dw1, db1, dGamma1, dBeta1] = conv_bn_relu_pool_backward(da1, cache1);
    
    if (computeDX), out = dX; grads = 0; return; end
    
    grads.w1 = dw1; grads.b1 = db1;
    grads.w2 = dw2; grads.b2 = db2;
    grads.w3 = dw3; grads.b3 = db3;
    grads.w4 = dw4; grads.b4 = db4;
    grads.w5 = dw5; grads.b5 = db5;
    
    grads.gamma1 = dGamma1; grads.beta1 = dBeta1;
    grads.gamma2 = dGamma2; grads.beta2 = dBeta2;
    grads.gamma3 = dGamma3; grads.beta3 = dBeta3;
    grads.gamma4 = dGamma4; grads.beta4 = dBeta4;
    
    opt.bnParam1 = bnParam1; opt.bnParam2 = bnParam2;
    opt.bnParam3 = bnParam3; opt.bnParam4 = bnParam4;
    
    regLoss = 0;
    if (weightDecay)
        W = {'w1', 'w2', 'w3', 'w4', 'w5'};
        for i = 1:numel(W)
            w = model.(W{i});
            regLoss = regLoss + 0.5 * weightDecay * sum(sum(sum(sum(w.^2))));
            grads.(W{i}) = grads.(W{i}) + weightDecay * w;
        end
    end
    out = dataLoss + regLoss;

end