function K1 = GetKernel1(X,Y)
K1 = zeros(size(X,1), size(Y,1));

for i = 1 : size(X,1)
   for j = 1 : size(Y,1)
      K1(i,j) = Kernel(X(i,:) , Y(j,:) , 'rbf'); 
   end
end
end

