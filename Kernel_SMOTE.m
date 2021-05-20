function [Kmat , [S;= Kernel_SMOTE(S,Y, P , k_knn)
Smin = S(Y==2,:);
Nmin = size(Smin,1);

S_seed = [];
S_neighbor = [];

for l = 1 : P
    Irand = randi(Nmin);
    xp = Smin(Irand,:);
    dist = GetKernalDist(Smin , xp);
    [~,Isort] = sort(dist);
    Isort = Isort(2 : k_knn+1);
    
    r = randi(length(Isort));
    
    xq = Smin(Isort(r) , :);
    
    S_seed = [S_seed ; xp];
    S_neighbor = [S_neighbor ; xq];
    
end

K1 = GetKernel1(S,S);
K2 = GetKernel2(S , S_seed , S_neighbor);
K3 = GetKernel3(S_seed , S_neighbor);

Kmat = [K1 , K2 ; K2' , K3];

end

