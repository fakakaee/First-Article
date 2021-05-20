function dist = GetKernalDist(X , Y)

dist = zeros(size(X,1) , size(Y,1));
for i = 1 : size(X,1)
    for j = 1: size(Y,1)
        dist(i,j) = Kernel(X(i,:) , X(i,:) , 'rbf') - ...
            2* Kernel(X(i,:) , X(j,:),'rbf') + ...
            Kernel(X(j,:) ,X(j,:),'rbf');
    end
end

end

