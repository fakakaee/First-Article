function ERR = Calc_Cluster_Error(Y , ptsC)
YY = Y;
Y(ptsC == 0) = [];
ptsC(ptsC == 0) = [];

numClusters = length(unique(ptsC));
ERR = zeros(numClusters ,1 );
WEIGHT = zeros(numClusters ,1 );
for i = 1 : numClusters
   Yi = Y(ptsC == i);
   
   if mode(Yi) == 1
        ERR(i) = sum(Yi ~= mode(Yi)) / length(Yi);
        ERR(i) = ERR(i) * (sum(YY == 1)/length(YY));
   else
       ERR(i) = sum(Yi ~= mode(Yi)) / length(Yi);
       ERR(i) = ERR(i) * (sum(YY == 2)/length(YY));
   end
   
   WEIGHT(i) = length(Yi);
end

ERR = sum(ERR .* WEIGHT) / sum(WEIGHT);
end

