function [model, pre_net] = funcDeepNetTrainNoRolling(train_data, train_target, cv_data,cv_label, test_data, test_label,model1,opts)
global is_wiener_mask;
%% network initialization
net_struct = opts.net_struct;
isGPU = 0;
fprintf(1,'1 is_wiener_mask=%d\n',is_wiener_mask);
if opts.isPretrain
    disp('start RBM pretraining...')
    pre_net = pretrainRBMStack(train_data,opts);
    disp('RBM pretraining done.')
else
    disp('use random weight initialization.')
    isSparse = 0; isNorm = 1;
    pre_net = randInitNet(net_struct,isSparse,isNorm,isGPU);    
end

net_iterative = pre_net;
num_net_layer = length(net_iterative);
num_samples = size(train_data,1);

batch_id = genBatchID(num_samples,opts.sgd_batch_size);

num_batch = size(batch_id,2);
fprintf('\nNum of Training Samples:%d\n',num_samples);
disp(net_struct);
disp(size(train_data));

net_weights_inc = zeroInitNet(net_struct, isGPU);
net_grad_ssqr = zeroInitNet(net_struct, isGPU, eps);
net_ada_eta = zeroInitNet(net_struct, isGPU);

cv_rec = repmat(struct,opts.sgd_max_epoch,1);
for epoch = 1:opts.sgd_max_epoch
    tic
    seq = randperm(num_samples); % randperm dataset every epoch
    cost_sum = 0;
    for bid = 1:num_batch-1
        perm_idx = seq(batch_id(1,bid):batch_id(2,bid));
  
            batch_data = train_data(perm_idx,:);
            batch_label = train_target(perm_idx,:);

        
        if epoch>opts.change_momentum_point;
            momentum=opts.final_momentum;
        else
            momentum=opts.initial_momentum;
        end
        
        %backprop: core code
        [cost,net_grad] = computeNetGradientNoRolling(net_iterative, batch_data, batch_label, opts);

        %supports only sgd
        for ll = 1:num_net_layer
            switch opts.learner
                case 'sgd'   % stochastic gradient descent
                    net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + opts.sgd_learn_rate(epoch)*net_grad(ll).W;
                    net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + opts.sgd_learn_rate(epoch)*net_grad(ll).b;
                case 'ada_sgd' % adaptive stochastic gradient descent
                    net_grad_ssqr(ll).W = net_grad_ssqr(ll).W + (net_grad(ll).W).^2;
                    net_grad_ssqr(ll).b = net_grad_ssqr(ll).b + (net_grad(ll).b).^2;
                    
                    net_ada_eta(ll).W = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).W);                    
                    net_ada_eta(ll).b = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).b);
                    
                    net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + net_ada_eta(ll).W.*net_grad(ll).W;
                    net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + net_ada_eta(ll).b.*net_grad(ll).b;
            end
            
            net_iterative(ll).W = net_iterative(ll).W - net_weights_inc(ll).W;    % update the weight for each layer
            net_iterative(ll).b = net_iterative(ll).b - net_weights_inc(ll).b;    % update the bias for each layer
        end
        cost_sum = cost_sum + cost;
    end
    fprintf('Objective cost at epoch %d: %2.2f \n', epoch, cost_sum);
    
    % check perf. on cv data
    if ~mod(epoch,opts.cv_interval)
         if is_wiener_mask == 0
             disp('binary mask');
             [perf, perf_str] = checkPerformanceOnData_no_print(net_iterative,cv_data,cv_label,opts);
         else
             disp('ratio mask');
             [perf, perf_str] = checkPerformanceOnData_no_print_wiener(net_iterative,cv_data,cv_label,opts);
         end
         cv_rec(epoch*opts.cv_interval).perf = perf;            % save the mse performance
         cv_rec(epoch*opts.cv_interval).model = net_iterative;  % save the model according to the epoch
    toc
    end
end

%% use the best model on dev_set
if is_wiener_mask == 0 
    % maximize HIT-FA tp get best model for estimated binary mask
    [m_v,m_i] = max([cv_rec.perf]);
else
    % minimize MSE to get best model for estimated wiener mask 
    % find the minimum value with all of the mse,  m_i the minimum iteration, the m_v is the minimum value
    [m_v,m_i] = min([cv_rec.perf]);  
end
model = cv_rec(opts.sgd_max_epoch).model;   % set the model as the best model for IRM
%use this model to predic on test set rather than dev set.
if is_wiener_mask == 0
    disp('IBM');
    [test_perf, test_perf_str] = checkPerformanceOnData_save_IBM(model,test_data,test_label,opts, 1);
else
    disp('wiener');
    [a, b] = checkPerformanceOnData_save_wiener(model,model1,test_data,test_label,opts, 1);
end
%use this model to predic on test set rather than dev set.

