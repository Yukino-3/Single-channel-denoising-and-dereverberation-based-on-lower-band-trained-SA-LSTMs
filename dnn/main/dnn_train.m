%% --setting net params--
opts.cv_interval = 1; % check cv perf. every this many epochs
opts.isPretrain = 0; % pre-training using RBM?
opts.rbm_max_epoch = 25; % max number of epoch in rbm
opts.rbm_batch_size = 187; % batch size for pretraining
opts.rbm_learn_rate_binary = 0.01;
opts.rbm_learn_rate_real = 0.004;

opts.learner = 'ada_sgd';    % 'ada_sgd' or 'sgd'
opts.sgd_max_epoch = 30;     % maximum number of training epochs
opts.sgd_batch_size = 1024;  % batch size for SGD
opts.ada_sgd_scale = 0.0015; % scaling factor for ada_grad
opts.sgd_learn_rate = linspace(0.05, 0.001, opts.sgd_max_epoch); % linearly decreasing lrate for plain sgd

opts.initial_momentum = 0.5;     % the initial momentum in ada_sdg
opts.final_momentum = 0.9;       % the final momentum in ada_sdg
opts.change_momentum_point = 5;  % the number of epoch where change the momentum

opts.cost_function = 'mse';      % type of cost function
opts.hid_struct = [1024 1024 1024]; % num of hidden layers and units of each layer

opts.unit_type_output = 'lin';  % type of activation function in output layer, sigm or relu
opts.unit_type_hidden = 'sigm'; % type of activation function in hidden layer, sigm or relu
% strcmp, compare the two input in the function
if strcmp(opts.unit_type_output,'softmax'); opts.cost_function = 'softmax_xentropy'; end;
opts.isDropout = 0;             % need dropout regularization?
opts.isDropoutInput = 0;        % dropout inputs?
opts.drop_ratio = 0.2;          % ratio of units to drop
opts.train_neighbour = 3;       % adjacent window length


opts.eval_on_gpu = 0;  
opts.save_on_fly = 0; % save the current best model along the way
opts.db = db;
opts.save_model_path = 'YOUR_FOLDER_OF_MODELS';
opts.note_str = 'YOUR_NOTES';

%% --network training--
% set final structure
[num_samples, dim_input] = size(test_data);     % obtain the number of samples and dimension of input 
dim_output = size(test_label, 2);               % dimension of output
opts.net_struct = [dim_input*(2*opts.train_neighbour+1), opts.hid_struct, dim_output];  % construct the structure of NN
opts 

% main training function
tic    % record running time
test_data = make_window_buffer(test_data, opts.train_neighbour);   % using slide window to obtain the adjacent frames
cv_data = make_window_buffer(cv_data, opts.train_neighbour);
train_data = make_window_buffer(train_handle.train_data, opts.train_neighbour);
train_target = train_handle.train_target;  
train_target1 = train_handle.train_target1;
train_mix = train_handle.train_mix;
% start to train model
% train first model for DM
[model1, pre_net1] = funcDeepNetTrainNoRolling1(train_data, train_target1, cv_data,cv_label1, test_data, test_label1,test_label,opts);
output1 = getOutputFromNetSplit(model1,train_data,5,opts);  % generate the output with trained model 2 and test data
output1 = invLogisticFunction(output1,10,1);
     
train_de = output1.*train_mix;
new_mask = train_target./train_de;
idx_inf=isnan(new_mask)| isinf(new_mask);
[line, column] = size(idx_inf);   
new_mask = zeros(line, column);
    for i=1:line
        for j=1:column
            new_mask(i,j)= sqrt(train_target(i,j)/ train_de(i,j) );
        end
    end

 new_mask = abs(new_mask);

     for i=1:line
         for j=1:column
             if idx_inf(i,j) == 1
                 new_mask(i,j) =0.1;
             end                
         end
     end

% train second model  for IRM
[model, pre_net] = funcDeepNetTrainNoRolling(train_data, new_mask, cv_data,cv_label, test_data, test_label,model1, opts);
train_time = toc;
fprintf('\nTraining done. Elapsed time: %2.2f sec\n',train_time);
