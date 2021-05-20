clc
clear
close all
load E2.mat
X = E2(: , 1:end-1);
Y = E2(: , end);
X = Normalize(X);
K1 = 10;
A = zeros (100,2);
n = 0 ;
MAX_ITER = 100;

Acc_WELM = zeros(K1,MAX_ITER);
GM_WELM  = zeros(K1,MAX_ITER);
for iter  = 1: MAX_ITER
    
    cvidx1 = crossvalind('Kfold' , size(X,1) , K1);

for kfold1 = 1 : K1
    
    [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx1 , kfold1);
    
    IncPer = 100*sum(Ytr==1)/sum(Ytr==2);
    XSMOTE = mySMOTE(Xtr(Ytr==2,:) , IncPer , 3);
    
    
    Xtr_balance = [Xtr(Ytr==1,:) ; XSMOTE];
    Ytr_balance = [ones(sum(Ytr==1),1) ; 2*ones(size(XSMOTE,1),1)];
    
    Xtr = Xtr_balance;
    Ytr = Ytr_balance;
    
    Model_WELM = WELM([2*Xtr-1 , Ytr]);
        Yts_predicted = elmpredict(Model_WELM,2*Xts-1);
        
        [Acc_WELM(kfold1,iter) , GM_WELM(kfold1,iter)] = GetEvaluationMetrics(Yts , Yts_predicted);
        
        if mod(iter,10) == 1
            disp(['WELM Accuracy is: ',num2str(Acc_WELM(kfold1)), '%']);
            disp(['WELM GM is: ',num2str(GM_WELM(kfold1)), '%']);
            disp('_________________________________');
        end
        
     n = 10 ;
        A(kfold1,n) = Acc_WELM(kfold1);
        n = n+1 ;
        A(kfold1, n) = GM_WELM(kfold1);
    end
end
filename = 'RWELM.xlsx';
xlswrite (filename,A);

disp(['WELM Average Accuracy is: ',num2str(mean(Acc_WELM(:))), '%']);
disp(['WELM Average STD Accuracy is: ',num2str(std(Acc_WELM(:)))]);

disp(['WELM Average GM is: ',num2str(mean(GM_WELM(:))), '%']);
disp(['WELM Average STD GM is: ',num2str(std(GM_WELM(:)))]);

save Model_WELM Model_WELM
