function [ mask ] = wiener( mat_s, mat_n )
    [line, column] = size(mat_s);
    mask = zeros(line, column);
    for i=1:line
        for j=1:column
            mask(i,j) =  mat_s(i,j);
        end
    end

end
