clc
clear
close all
load Data_Imp_Linear.mat
% X = E2(: , 1:end-1);
% Y = E2(: , end);
X = Normalize(X);
K1 = 10;
A = zeros (100,2);
n = 0 ;

MAX_ITER = 100;

Acc_WELM = zeros(K1,MAX_ITER);
GM_WELM  = zeros(K1,MAX_ITER);
for iter  = 1: MAX_ITER
    
    cvidx = crossvalind('Kfold' , size(X,1) , K1);
    
    
    for kfold = 1 : K1
        
        [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx , kfold);
        
        Model_WELM = WELM([2*Xtr-1 , Ytr]);
        Yts_predicted = elmpredict(Model_WELM,2*Xts-1);
        
        [Acc_WELM(kfold,iter) , GM_WELM(kfold,iter)] = GetEvaluationMetrics(Yts , Yts_predicted);
        
        if mod(iter,10) == 1
            disp(['WELM Accuracy is: ',num2str(Acc_WELM(kfold)), '%']);
            disp(['WELM GM is: ',num2str(GM_WELM(kfold)), '%']);
            disp('_________________________________');
        end
        
        n = 5 ;
        A(kfold,n) = Acc_WELM(kfold);
        n = n+1 ;
        A(kfold, n) = GM_WELM(kfold);
    end
end
filename = 'RWELM.xlsx';
xlswrite (filename,A);


disp(['WELM Average Accuracy is: ',num2str(mean(Acc_WELM(:))), '%']);
disp(['WELM Average STD Accuracy is: ',num2str(std(Acc_WELM(:)))]);

disp(['WELM Average GM is: ',num2str(mean(GM_WELM(:))), '%']);
disp(['WELM Average STD GM is: ',num2str(std(GM_WELM(:)))]);

save Model_WELM Model_WELM
