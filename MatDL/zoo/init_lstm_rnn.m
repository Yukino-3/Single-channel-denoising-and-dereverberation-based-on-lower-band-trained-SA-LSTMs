function [model, opt] = init_lstm_rnn(N, K, layers_size, opt)  % N input dimension, K class, hidden layer size
    
    wieghtScale = 1;
    biasScale = 0;
    
    WLSTM = randn(N + layers_size(1) + 1, 4 * layers_size(1)) / sqrt(N + layers_size(1));  % initial weights
    WLSTM2 = randn(layers_size(1) + layers_size(2) + 1, 4 * layers_size(2)) / sqrt(layers_size(1) + layers_size(2));
    
    WLSTM(1, :) = 0; % initialize biases to zero
    WLSTM2(1, :) = 0; % initialize biases to zero
    
    fancy_forget_bias_init = 3;
    if fancy_forget_bias_init ~= 0
        % forget gates get little bit negative bias initially to encourage them to be turned off
        % remember that due to Xavier initialization above, the raw output activations from gates before
        % nonlinearity are zero mean and on order of standard deviation ~1
        WLSTM(1, (layers_size(1) + 1) : (2 * layers_size(1))) = fancy_forget_bias_init;
        WLSTM2(1, (layers_size(2) + 1) : (2 * layers_size(2))) = fancy_forget_bias_init;
    end
    model.WLSTM = WLSTM;
    model.WLSTM2 = WLSTM2;
    model.wy = randn(layers_size(2), K) * wieghtScale;  % weights of output, layers_size(2) is the number of hidden units
    model.by = randn(1, K) * biasScale;                 % bias of output  
    
    p = fieldnames(model);
    for i = 1:numel(p)  % numel : number of elements in p
        opt.vgrads.(p{i}) = zeros(size(model.(p{i})));
    end

end