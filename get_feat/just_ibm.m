function [ features, ideal_mask,mixture] = just_ibm( mix_sig, voice_sig, noise_sig,fun_name, NUMBER_CHANNEL,is_wiener_mask, db)
SAMPLING_FREQUENCY = 16000;
WINDOW     = (SAMPLING_FREQUENCY/50);     % use a window of 20 ms */
OFFSET     = (WINDOW/2);                  % compute acf every 10 ms */

[n_sample, m] = size(mix_sig);

n_frame = floor(n_sample/OFFSET);   % Number of frames
% Produce an array of filtered responses from a Gammatone filterbank
g_voice = gammatone(double(voice_sig), NUMBER_CHANNEL, [50 8000], SAMPLING_FREQUENCY);  
% Generate a cochleagram from responses of a Gammatone filterbank.
coch_voice = cochleagram(g_voice, 320);
g_noise = gammatone(double(noise_sig), NUMBER_CHANNEL, [50 8000], SAMPLING_FREQUENCY);
coch_noise = cochleagram(g_noise, 320);

if is_wiener_mask == 0
    ideal_mask = ideal(coch_voice, coch_noise, db);   % Binary Mask
else
    ideal_mask = wiener(coch_voice, coch_noise);      % Soft Mask
end
mixture = coch_voice+ coch_noise;
f1 = feval(fun_name,mix_sig);      % Obtain the feature combination

features(:,:) = f1(:,1:n_frame);

features = single(features);
ideal_mask = single(ideal_mask);
mixture = single(mixture);
end
