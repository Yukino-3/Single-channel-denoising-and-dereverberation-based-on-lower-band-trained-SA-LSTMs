function [a,b] = checkPerformanceOnData_save_wiener(net,net1,data,label,opts,write_wav,num_split)
disp('save_wiener_func');
global feat noise frame_index DFI;
global small_mix_cell small_noise_cell small_speech_cell;
num_test_sents = size(DFI,1)    % number of testing data
Fs = 16e3;
if nargin < 6
    num_split = 1;
end
a = 1;
b = 2;
num_samples = size(data,1);

if ~opts.eval_on_gpu
    for i = 1:length(net)
        net(i).W = gather(net(i).W);
        net(i).b = gather(net(i).b);
        data = gather(data);
    end
end
% NEW MASK
newmask = getOutputFromNetSplit(net,data,5,opts);  % generate the output with trained model 1 and test data
 
 % dm
 output1 = getOutputFromNetSplit(net1,data,5,opts);  % generate the output with trained model 2 and test data
 %% Derever
  % decompress DM
  output1 = invLogisticFunction(output1,10,1);
  % generate DM*IRM
output = output1 .*newmask;

%%
est_r = cell(num_test_sents);
ideal_r = cell(num_test_sents);
clean_s = cell(num_test_sents);
mix_s = cell(num_test_sents);
EST_MASK = cell(num_test_sents);
IDEAL_MASK = cell(num_test_sents);
stoi_est_sum = 0;
PESQ_est_sum = 0;
SNR_est_sum = 0;
improved_SDR_sum = 0;
unprocessed_SNR_sum = 0;
unprocessed_stoi_sum = 0;
unprocessed_PESQ_sum = 0;
% ideal_stoi_sum = 0;
noise_feat = sprintf('%-15s', [noise ' ' feat]);
for i=1:num_test_sents
    EST_MASK{i} = transpose(output(DFI(i,1):DFI(i,2),:));
    IDEAL_MASK{i} = transpose(label(DFI(i,1):DFI(i,2),:));
    mix = double(small_mix_cell{i});
    mix_s{i} = mix;
    % recover speech signal
    est_r{i} = synthesis(mix, double(EST_MASK{i}), [50, 8000], 320, 16e3);   % reconstruct estimated signals
    
    ideal_r{i} = synthesis(mix, double(IDEAL_MASK{i}), [50, 8000], 320, 16e3);% reconstruct ideal signals

    clean_s{i} = double(small_speech_cell{i});
    % caculate STOI with estimated speech signal
    est_stoi = stoi(clean_s{i}, est_r{i}, 16e3);
    % caculate the STOI of mixture
    unprocessed_stoi = stoi(clean_s{i}, mix, 16e3);


    
    %% -------- Initialize variables ----------- %
sil_sec           = zeros(1.5*Fs,1);
min_speech_length = 8; % minimum length of signal for PESQ (in seconds)
num_times_2_cat   = ceil((min_speech_length*Fs)/length(mix_s{i}));


%% ---------- Process signals for PESQ calculation ------------ %
cs  = [repmat([sil_sec;clean_s{i}],num_times_2_cat,1);sil_sec];
rs  = [repmat([sil_sec;mix_s{i}],num_times_2_cat,1);sil_sec];
drs = [repmat([sil_sec;est_r{i}'],num_times_2_cat,1);sil_sec];
    %% ------------- Compute PESQ ----------------- %
    unprocessed_PESQ  = pesq_dat(cs,rs,Fs);  % PESQ
    if isnan(unprocessed_PESQ)
        unprocessed_PESQ = [];
    end
        est_PESQ  = pesq_dat(cs,drs,Fs); % PESQ
    if isnan(est_PESQ)
        est_PESQ = [];
    end


   %% -------------- BSS Toolkit ------------------------------ %
   % calculate SDR of mixture
    [xTarget, xeInterf, xeArtif]                     = bss_decomp_gain(mix_s{i}.', 1, [clean_s{i} (mix_s{i}-clean_s{i})].');
    [SDR_rev, SIR_rev, SAR_rev] = bss_crit(xTarget, xeInterf, xeArtif);
   % calculate SDR of estimated speech signal
    [yTarget, yeInterf, yeArtif]                           = bss_decomp_gain(est_r{i}, 1, [clean_s{i} (mix_s{i}-clean_s{i})].');
    [SDR_derev, SIR_derev, SAR_derev] = bss_crit(yTarget, yeInterf, yeArtif); 
    % calculate SDR improvement
    improved_SDR = SDR_derev - SDR_rev;
    % calculate SNR_fw of mixture
    unprocessed_SNR   = comp_fwseg(clean_s{i}, mix_s{i},Fs);
    % calculate SNR_fw of estimated speech signal
    est_SNR = comp_fwseg(clean_s{i}, est_r{i}',Fs);
    
    % calculate the averaged value of above measures
    improved_SDR_sum = improved_SDR_sum + improved_SDR;
    SNR_est_sum = SNR_est_sum + est_SNR;
    stoi_est_sum = stoi_est_sum + est_stoi;
    PESQ_est_sum = PESQ_est_sum + est_PESQ;
    unprocessed_SNR_sum = unprocessed_SNR_sum + unprocessed_SNR;
    unprocessed_stoi_sum = unprocessed_stoi_sum + unprocessed_stoi;
    unprocessed_PESQ_sum = unprocessed_PESQ_sum + unprocessed_PESQ;

     
end
fprintf(1,['\n#STOI_average# ' noise_feat ' unprocessed_STOI=%0.4f  est_STOI=%0.4f \n'], unprocessed_stoi_sum/num_test_sents, stoi_est_sum/num_test_sents);
fprintf(1,['\n#PESQ_average# ' noise_feat ' unprocessed_PESQ=%0.4f  est_PESQ=%0.4f \n'], unprocessed_PESQ_sum/num_test_sents, PESQ_est_sum/num_test_sents);
fprintf(1,['\n#SNR_average# ' noise_feat ' unprocessed_SNR=%0.4f  est_SNR=%0.4f \n'], unprocessed_SNR_sum/num_test_sents, SNR_est_sum/num_test_sents);
fprintf(1,['\n#Improved SDR_average# ' noise_feat ' Improved SDR=%0.4f \n'], improved_SDR_sum/num_test_sents);

