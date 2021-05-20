clc
clear
close all
load E2.mat
X = E2(: , 1:end-1);
Y = E2(: , end);
X = Normalize(X);

dist_type = 'euclidean'; % we can also use 'mahalanobis', 'correlation' , 'cosine' and etc


Dist_Minor_Minor = pdist2( X(Y ~= 1, : ) , X(Y ~= 1, : ) , dist_type);

Dist_Minor_Minor = Dist_Minor_Minor(Dist_Minor_Minor ~= 0);
sortedDist_minor_minor = sort(Dist_Minor_Minor);
disp(['min dist minor-minor: ',num2str(sortedDist_minor_minor(1))]);
disp(['max dist minor-minor : ',num2str(sortedDist_minor_minor(end))]);

figure
histogram(Dist_Minor_Minor(:));
title('Histogram of distances between minors (zeroes are eliminated)');


NumOutliers = zeros(length(sortedDist_minor_minor),1);
NumClusters = zeros(length(sortedDist_minor_minor),1);
NumCorePoints = zeros(length(sortedDist_minor_minor),1);
ErrorClusters = zeros(length(sortedDist_minor_minor),1);
minPts = 2;
STEP = 300;
for i = 1 :STEP : length(sortedDist_minor_minor)
    disp(['iter = ' ,num2str(i) ,' from ',num2str(length(sortedDist_minor_minor))]);
    
    EPS = sortedDist_minor_minor(i);
    [C , ptsC, centers] = dbscan(X' , EPS , minPts);
    
    NumOutliers(i) = sum(ptsC == 0);
    NumClusters(i) = size(centers , 2);
%     disp(['eps: ',num2str(EPS)]);
%     disp(['minPts: ',num2str(minPts)]);
    
%     for i = 1 : length(C)
%         disp(['center (',num2str(i),'): ',num2str(centers(:,i)')]);
%     end
    
    numcore = zeros(length(C),1);
    for j = 1 : length(C)
        numcore(j) = length(C{j});
%         disp(['number of core points in cluster (',num2str(j),'): ',num2str(numcore(j))]);
    end
%     disp(['number of all core points: ',num2str(sum(numcore))]);
    
    NumCorePoints(i) = sum(numcore);
    
    ErrorClusters(i) = Calc_Cluster_Error(Y , ptsC);
end

figure
plot(sortedDist_minor_minor(1:STEP:end) , NumCorePoints(1:STEP:end),'linewidth',1.8);
grid on
title('Impact of Eps on Num Core Points');
xlabel('eps');
ylabel('num core points');


figure
plot(sortedDist_minor_minor(1:STEP:end) , NumClusters(1:STEP:end),'linewidth',1.8);
grid on
title('Impact of Eps on Num Clusters');
xlabel('eps');
ylabel('num clusters');

figure
plot(sortedDist_minor_minor(1:STEP:end) , NumOutliers(1:STEP:end),'linewidth',1.8);
grid on
title('Impact of Eps on Num Outliers');
xlabel('eps');
ylabel('num outlier');

figure
plot(sortedDist_minor_minor(1:STEP:end) , ErrorClusters(1:STEP:end),'linewidth',1.8);
grid on
title('Impact of Eps on Cluster Errors');
xlabel('eps');
ylabel('clustering error')