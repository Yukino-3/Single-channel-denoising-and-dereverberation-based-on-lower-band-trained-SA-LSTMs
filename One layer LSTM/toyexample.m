% This is a toy example of the use of the code, and the network created has
% only a lstm layer
clear
clc
%% set parameters for the lstmcell
seq_len = 11;
len_in = 245;
len_hidden = 1024;
len_hidden1 = 1024;
len_out = 64;

active_funcs = {'sigm', 'sigm'};
opt.learningRate = 0.1;
opt.weightPenaltyL2 = 0.001;
opt.momentum = 0.5;
opt.scaling_learningRate = 0.5;


x = rand(seq_len, len_in+1);
y = rand(seq_len, len_out);
lstmcell = lstmcellsetup(len_in, len_hidden, opt, active_funcs);
lstmcell1 = lstmcellsetup(len_hidden-1, len_hidden1, opt, active_funcs);
lstmcell2 = lstmcellsetup(len_hidden1-1, len_out, opt, active_funcs);
lstmcell = lstmcellff1(lstmcell, x);
lstmcell1 = lstmcellff1(lstmcell1, lstmcell.mh);
lstmcell2 = lstmcellff(lstmcell2, lstmcell1.mh, y);
e = y - lstmcell2.mh;
loss_1 = sum(sum(e .* e)) / 2 / seq_len;

%%

%% train the network
for i = 1:100
    lstmcell = lstmcellff1(lstmcell, x);
    lstmcell1 = lstmcellff1(lstmcell1, lstmcell.mh);
    lstmcell2 = lstmcellff(lstmcell2, lstmcell1.mh, y);
    e = y - lstmcell2.mh;
     e1 = lstmcell2.x - lstmcell1.mh;
     e2 = lstmcell1.x - lstmcell.mh;
    loss(i) = sum(sum(e .* e)) / 2 / seq_len;
    loss1(i) = sum(sum(e1 .* e1)) / 2 / seq_len;
    loss2(i) = sum(sum(e2 .* e2)) / 2 / seq_len;
    lstmcell2 = lstmcellbp(lstmcell2, -e);
    lstmcell1 = lstmcellbp(lstmcell1, -e1);
    lstmcell = lstmcellbp(lstmcell, -e2);
    lstmcell2 = lstmcellupdate(lstmcell2);
    lstmcell1 = lstmcellupdate(lstmcell1);
    lstmcell = lstmcellupdate(lstmcell);  
end
plot(loss);
%plot(loss1);
% plot(loss2);