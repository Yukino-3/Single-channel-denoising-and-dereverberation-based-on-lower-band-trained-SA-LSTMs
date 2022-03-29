

%% This method can increase the performance of the IRM slightly, but it needs
%  further research to improve the performance 
%%
[line, column] = size(coch_noise);
    mask = zeros(line, column);
    for i=1:line
        for j=1:column
            if coch_noise(i,j) >= 100*coch_voice(i,j) 
                mask(i,j) = 0;
            else
                mask(i,j) = 1;
            end
        end
    end
%%