function [acc , gm] = GetEvaluationMetrics(Y1 , Y2)
acc = sum(Y1 == Y2) / length(Y1);

K = length(unique(Y1));
Sensivisity = zeros(K,1);
for k = 1 : K
   Nk = sum(Y1 == k);
   Sensivisity(k) = sum(Y1(Y1 == k) == Y2(Y1==k) ) / Nk;
end

gm = prod(Sensivisity)^(1/K);


end

