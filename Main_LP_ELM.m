clc
clear
close all
load Data_Imp_Linear.mat
% X = E2(: , 1:end-1);
% Y = E2(: , end);
X = Normalize(X);
K1 = 10;

A = zeros (100,2);
n=0;

MAX_ITER = 100;

Acc_ELM = zeros(K1,MAX_ITER);
GM_ELM  = zeros(K1,MAX_ITER);
for iter  = 1: MAX_ITER
    
    cvidx1 = crossvalind('Kfold' , size(X,1) , K1);
    
    
    for kfold = 1 : K1
        
        [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx1 , kfold);
        L = sum(Ytr == 2);
        Donor_idx = [5 , 6 ,15];
        Recipient_idx = setdiff(1 : size(Xtr,2) , Donor_idx);
        
        Xdon = X(: , Donor_idx);
        Xrec = X(: , Recipient_idx);
        
        Xpot = zeros(L*(L-1) , size(X,2));
        cnt = 1;
        for i = 1: L
            for j = 1 : L
                if i == j
                    continue
                end
                Xpot(cnt , Donor_idx) = Xtr(i , Donor_idx);
                Xpot(cnt , Recipient_idx) = Xtr(j , Recipient_idx);
                cnt = cnt + 1;
            end
        end
        
        k = 3;
        Model_KNN = fitcknn(Xtr , Ytr,'NumNeighbors',k);
        Ypot = predict(Model_KNN, Xpot);
        
        Xtr_balance = [Xtr ; Xpot(Ypot == 2 ,:)]; % to do: check if minority class be too large
        Ytr_balance = [Ytr ; Ypot(Ypot == 2)];
        
        Xtr = Xtr_balance;
        Ytr = Ytr_balance;
        
        Model_ELM = ELMClassifier([2*Xtr_balance-1 , Ytr_balance]);
        
        Yts_predicted = elmpredict(Model_ELM,2*Xts-1);
        
        [Acc_ELM(kfold,iter) , GM_ELM(kfold,iter)] = GetEvaluationMetrics(Yts , Yts_predicted);
        
        if mod(iter,10) == 1
            disp(['ELM Accuracy is: ',num2str(Acc_ELM(kfold)), '%']);
            disp(['ELM GM is: ',num2str(GM_ELM(kfold)), '%']);
            disp('_________________________________');
        end
      n = 1;
        A(kfold,n) = Acc_ELM(kfold);
        n = n+1 ;
        A(kfold, n) = GM_ELM(kfold);
    end
end
filename = 'RWELM.xlsx';
xlswrite (filename,A);
  
disp(['ELM Average Accuracy is: ',num2str(mean(Acc_ELM(:))), '%']);
disp(['ELM Average STD Accuracy is: ',num2str(std(Acc_ELM(:)))]);

disp(['ELM Average GM is: ',num2str(mean(GM_ELM(:))), '%']);
disp(['ELM Average STD GM is: ',num2str(std(GM_ELM(:)))]);

save Model_ELM Model_ELM