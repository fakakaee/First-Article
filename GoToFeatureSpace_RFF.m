function PHX = GoToFeatureSpace_RFF(X,R)

R = R / 2;

d = size(X,2);
w = randn(d , R);

PHX = zeros(size(X,1) , 2*R);

for i = 1 : size(X,1)
    XX = X(i,:) * w;
    PHX(i,:) = (1/sqrt(R)) * [cos(XX) , sin(XX)];
end


end

