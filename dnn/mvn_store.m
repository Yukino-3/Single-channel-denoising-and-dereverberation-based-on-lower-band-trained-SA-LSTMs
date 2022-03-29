function run_every(noise, feat, db, TMP_STORE)
format compact;

fprintf(1,'MVNing Feat=%s Noise=%s\n', feat, noise);

root_path = [TMP_STORE filesep 'db' num2str(db) filesep]
speech_data_path = [root_path 'mix' filesep 'test_' noise '_mix_aft2.mat'];
load(speech_data_path);

% test_set
load([root_path 'feat' filesep 'test_' noise '_' feat '.mat']); %test set
test_data = feat_data; test_label = feat_label; test_label1 = feat_label1; 
clear feat_data feat_label feat_label1

% train_set
train_path = [root_path 'feat' filesep 'train_' noise '_' feat '.mat']

train_data = []; train_target = []; train_target1 = [];train_mix = [];
disp(['loading ' train_path]);
load(train_path);
train_data = [train_data ; feat_data];         % load the data for input
train_mix = [train_mix ; feat_mix];
train_target = [train_target ; feat_label];    % load the traininig target for IRM
train_target1 = [train_target1 ; feat_label1]; % load the traininig target for DM
clear feat_data feat_label feat_label1 feat_mix;
% cv means cross validation
cv_portion = floor(0.1 * size(train_data, 1)); % select part of data for validation

fprintf(1,'Total=%d, cv=%d  train=%d\n',size(train_data, 1), cv_portion, size(train_data, 1) - cv_portion);
cv_data = train_data(1:cv_portion,:);
cv_label = train_target(1:cv_portion,:);
cv_label1 = train_target1(1:cv_portion,:);

train_data(1:cv_portion,:) = [];
train_target(1:cv_portion,:) = [];
train_target1(1:cv_portion,:) = [];
train_mix(1:cv_portion,:) = [];

[a2, b2] = size(cv_data);
[a3, b3] = size(test_data);
fprintf(1,'cv=%d x %d, test=%d x %d\n',a2,b2,a3,b3);

[train_data,para.tr_mu,para.tr_std] = mean_var_norm(train_data);
cv_data = mean_var_norm_testing(cv_data, para.tr_mu,para.tr_std);
test_data = mean_var_norm_testing(test_data, para.tr_mu,para.tr_std);

save_mvn_prefix_path = ['MVN_STORE' filesep];
if ~exist(save_mvn_prefix_path,'dir'); mkdir(save_mvn_prefix_path); end;
MVN_DATA_PATH = [save_mvn_prefix_path 'allmvntrain_' noise '_' feat '_' num2str(db) '.mat']
save(MVN_DATA_PATH, 'train_data','train_target','train_target1','train_mix','cv_data','cv_label','cv_label1','test_data','test_label','test_label1', 'DFI',...
 'small_mix_cell', 'small_noise_cell', 'small_speech_cell', 'c_mat', '-v7.3');%also saved test mixes

pause(1);

end
