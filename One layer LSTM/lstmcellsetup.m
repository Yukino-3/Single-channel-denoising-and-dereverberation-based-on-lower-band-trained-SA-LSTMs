function lstmcell = lstmcellsetup(inputlen, outputlen, opt, active)
% lstmcellsetup create a lstmcell layer for a Feedforword Backpropagate Neural Network 
% and it can be used as a active function conveniently.
% Notice that this setup will add biases automaticly.
% inputlen:   length of input layer
% outputlen:  length of output layer
% opt:        parameters for network optimize    
% active:     active function for the network
%% lstmcell setup
    lstmcell.inputlen = inputlen;
    lstmcell.outputlen = outputlen;
    lstmcell.delta = active{1}; % active function for gate
    lstmcell.g = active{2}; % active function for c
    %lstmcell.h = 'linear';
    lstmcell.learningRate = opt.learningRate;
    lstmcell.momentum = opt.momentum;
    lstmcell.weightPenaltyL2 = opt.weightPenaltyL2;
    lstmcell.scaling_learningRate = opt.scaling_learningRate;

% bias_input_gate=rand(1,cell_num);
% bias_forget_gate=rand(1,cell_num);
% bias_output_gate=rand(1,cell_num);
%% initialization of weights
    %i_t
    lstmcell.W_ix = (rand(outputlen, inputlen + 1) - 0.5) / inputlen;  % input gate weight with input,x
    lstmcell.W_ih = (rand(outputlen, outputlen) - 0.5) / outputlen;    % input gate weight with state,(h-1)
    lstmcell.W_ic = (rand(outputlen, 1) - 0.5) / outputlen; % diagonal matrix represent by vertor input cell
    
    %f_t
    lstmcell.W_fx = (rand(outputlen, inputlen + 1) - 0.5) / inputlen;  % forget gate with input x
    lstmcell.W_fh = (rand(outputlen, outputlen) - 0.5) / outputlen;    % forget gate with previous status
    lstmcell.W_fc = (rand(outputlen, 1) - 0.5) / outputlen; % diagonal matrix represent by vertor forget cell
    
    %c_t
    lstmcell.W_cx = (rand(outputlen, inputlen + 1) - 0.5) / inputlen;  % cell weight with input
    lstmcell.W_ch = (rand(outputlen, outputlen) - 0.5) / outputlen;    % cell weight with previous status
    
    %o_t
    lstmcell.W_ox = (rand(outputlen, inputlen + 1) - 0.5) / inputlen;  % output gate weight with input x
    lstmcell.W_oh = (rand(outputlen, outputlen) - 0.5) / outputlen;    % output gate weight with previous status
    lstmcell.W_oc = (rand(outputlen, 1) - 0.5) / outputlen; % diagonal matrix represent by vertor
    
    if lstmcell.momentum  > 0
        lstmcell.vW_ix = zeros(size(lstmcell.W_ix));
        lstmcell.vW_ih = zeros(size(lstmcell.W_ih));
        lstmcell.vW_ic = zeros(size(lstmcell.W_ic));

        lstmcell.vW_fx = zeros(size(lstmcell.W_fx));
        lstmcell.vW_fh = zeros(size(lstmcell.W_fh));
        lstmcell.vW_fc = zeros(size(lstmcell.W_fc));


        lstmcell.vW_cx = zeros(size(lstmcell.W_cx));
        lstmcell.vW_ch = zeros(size(lstmcell.W_ch));


        lstmcell.vW_ox = zeros(size(lstmcell.W_ox));
        lstmcell.vW_oh = zeros(size(lstmcell.W_oh));
        lstmcell.vW_oc = zeros(size(lstmcell.W_oc ));
    end
    
end