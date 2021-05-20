clear
close all
load Data_Imp_Linear.mat
% X = E2(: , 1:end-1);
% Y = E2(: , end);
X = Normalize(X);
K1 = 10;

n=0;
A = zeros (100,2);
MAX_ITER = 100;

Acc_ELM = zeros(K1,MAX_ITER);
GM_ELM  = zeros(K1,MAX_ITER);
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
    
    Model_ELM = ELMClassifier([2*Xtr_balance-1 , Ytr_balance]);
        
        Yts_predicted = elmpredict(Model_ELM,2*Xts-1);
        
        [Acc_ELM(kfold1,iter) , GM_ELM(kfold1,iter)] = GetEvaluationMetrics(Yts , Yts_predicted);
        
        if mod(iter,10) == 1
            disp(['ELM Accuracy is: ',num2str(Acc_ELM(kfold1)), '%']);
            disp(['ELM GM is: ',num2str(GM_ELM(kfold1)), '%']);
            disp('_________________________________');
        end
        n =1 ;
        A(kfold1,n) = Acc_ELM(kfold1);
        n = n+1 ;
        A(kfold1, n) = GM_ELM(kfold1);
    end
end
filename = 'result-16May.xlsx';
xlswrite (filename,A);



disp(['ELM Average Accuracy is: ',num2str(mean(Acc_ELM(:))), '%']);
disp(['ELM Average STD Accuracy is: ',num2str(std(Acc_ELM(:)))]);

disp(['ELM Average GM is: ',num2str(mean(GM_ELM(:))), '%']);
disp(['ELM Average STD GM is: ',num2str(std(GM_ELM(:)))]);

save Model_ELM Model_ELM