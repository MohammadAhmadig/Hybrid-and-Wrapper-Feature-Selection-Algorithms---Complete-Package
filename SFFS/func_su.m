function [ dC ] = func_su( a,b )
    [m,~] = size(b);
    dC = zeros(m,1);
    for ii=1:m
        dC(ii) = (h(a') + h(b(ii,:)'))/(2*mi(a',b(ii,:)'));% distance with SU (inverse SU)
    end
end
