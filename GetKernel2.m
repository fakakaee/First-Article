function K2 = GetKernel2(S , S_seed , S_neighbor)


K2 = zeros(size(S,1), size(S_seed,1));

del_pq = rand(size(S_seed,1),1);

for i = 1 : size(S,1)
   for j = 1 : size(S_seed,1)
       del = del_pq(j);
      K2(i,j) = (1-del)*Kernel(S(i,:) , S_seed(j,:) , 'rbf')...
          + del * Kernel(S(i,:) , S_neighbor(j,:) , 'rbf');
   end
end


end

