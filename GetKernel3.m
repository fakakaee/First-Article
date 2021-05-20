function K3 = GetKernel3( S_seed , S_neighbor)


K3 = zeros(size(S_seed,1), size(S_neighbor,1));

for i = 1 : size(S_seed,1)
   for j = 1 : size(S_neighbor,1)
       del = rand;
      K3(i,j) = (1-del)*(1-del)*Kernel(S_seed(i,:),S_seed(j,:),'rbf')+...
          (1-del)*(1-del)*Kernel(S_seed(i,:),S_neighbor(j,:),'rbf')+...
          (1-del)*(1-del)*Kernel(S_neighbor(i,:),S_seed(j,:),'rbf')+...
          (1-del)*(1-del)*Kernel(S_neighbor(i,:),S_neighbor(j,:),'rbf');
   end
end


end

