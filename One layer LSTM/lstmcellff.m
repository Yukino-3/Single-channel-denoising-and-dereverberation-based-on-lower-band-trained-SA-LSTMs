function cell = lstmcellff(cell, x, y)
    %LSTMFF performs a feedforward pass
    %   lstmff(lstmcell, x, y)

    [m, n] = size(x);
    if(n ~= cell.inputlen + 1)
        error('Error!!!  Input lengh is not correspond with the lstm setup!')
    end
    mi = zeros(m, cell.outputlen);  % memory of the input gate 
    mai = zeros(m, cell.outputlen); % memory of the input gate's sumation 
    mf = zeros(m , cell.outputlen); % memory of the forget gate
    maf = zeros(m, cell.outputlen); % memory of the forget gate's sumation
    mc = zeros(m, cell.outputlen);  % memory of the cell
    mac = zeros(m, cell.outputlen); % memory of the cell's sumation
    mgac = zeros(m, cell.outputlen);% memory of the activation computed by current input
    mo = zeros(m, cell.outputlen);  % memory of the output gate
    mao = zeros(m, cell.outputlen); % memory of the output gate's sumation
    mgc = zeros(m, cell.outputlen); % memory of the activation computed by current input and history
    mh = zeros(m, cell.outputlen);  % memory of the output(hidden layer output)

    %%comput the memory at time 1
    mai(1,:) =  x(1,:) * cell.W_ix';  % W_i*x_t, where t is 1
    mi(1,:) = active_func(mai(1,:), cell.delta);  % i_t, after activation function

    maf(1,:) = x(1,:) * cell.W_fx';   % Weight of forget gate, W_f*x_t at first time frame
    mf(1,:) = active_func(maf(1,:), cell.delta);   % f_t, after activation function

    mac(1,:) = x(1,:) * cell.W_cx';  % w_c*x_i
    mgac(1,:) = active_func(mac(1,:), cell.g);   % c~_t, the activation function should be tanh
    mc(1,:) = mi(1,:) .*  mgac(1,:);   % c_t = activation function of input * c~_t

    mao(1,:) = x(1,:) * cell.W_ox' + mc(1,:) .* cell.W_oc';  % x_t*W_o +c_t*output's weight
    mo(1,:) = active_func(mao(1,:), cell.delta);             % output of output gate, c_t
    
    mgc(1, :) = active_func(mc(1,:), cell.g);                % c_t with activation function
    mh(1,:) = mo(1,:) .* mgc(1, :);                          % c_t*output, h_t

%% compute memory for each time    
    for t = 2 : m
        % a_i(t) = W_ix * x(t) + W_ih * h(t-1) + W_ic * c(t-1)
        mai(t,:) =  x(t,:) * cell.W_ix' + mh(t-1, :) * cell.W_ih' + mc(t-1, :) .* cell.W_ic'; 
        mi(t,:) = active_func(mai(t,:), cell.delta); % input gate

        % a_f(t) = W_fx * x(t) + W_fh * h(t-1) + W_fc * c(t-1)
        maf(t,:) = x(t,:) * cell.W_fx' + mh(t-1, :) * cell.W_fh' + mc(t-1, :) .* cell.W_fc';
        mf(t,:) = active_func(maf(t,:), cell.delta); % forget gate

        % a_c(t) = W_ci * x(t) + W_ch * h(t-1)
        mac(t,:) = x(t,:) * cell.W_cx' +  mh(t-1, :) * cell.W_ch'; % cell
        % gac(t) = g(a_c(t))
        mgac(t,:) = active_func(mac(t,:), cell.g);   %c~_t
        % c(t) = f(t) * c(t-1) + i(t) * gac(t)
        mc(t,:) = mf(t,:) .* mc(t-1, :) + mi(t,:) .*  mgac(t,:); % c_t = [c_(t-1)*f(t)]+[i(t)*c~_(t)]

        % a_o(t) = W_ox * x(t) + W_oh * h(t-1) + W_oc * c(t)  input, before
        % activation function in output gate
        mao(t,:) = x(t,:) * cell.W_ox' + mh(t-1, :) * cell.W_oh' + mc(t,:) .* cell.W_oc';
        % o(t) = delta(a_o(t))
        mo(t,:) = active_func(mao(t,:), cell.delta); % output gate
        
        % gc(t) = g(c(t))
        mgc(t, :) = active_func(mc(t,:), cell.g);  % c_t after activation function
        % h(t) = o(t) * gc(t)
        mh(t,:) = mo(t,:) .* mgc(t, :);  % h_t = o_t * tanh(c_t), final output of h_t
    end
%% 
    cell.x = x;
    cell.mi = mi;
    cell.mai = mai;
    cell.mf = mf;
    cell.maf = maf;
    cell.mc = mc;
    cell.mac = mac;
    cell.mgac = mgac;
    cell.mo = mo;
    cell.mao = mao;
    cell.mgc = mgc;
    cell.mh = mh;
end