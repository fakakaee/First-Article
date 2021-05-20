clc
clear
close all
load E2.mat
X = E2(: , 1:end-1);
Y = E2(: , end);
X = Normalize(X);
K1 = 10;

MAX_ITER = 1;

Acc_WELM = zeros(K1,MAX_ITER);
GM_WELM  = zeros(K1,MAX_ITER);
for iter  = 1: MAX_ITER
    
    cvidx = crossvalind('Kfold' , size(X,1) , K1);
    
    
    for kfold = 1 : K1
        
        [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx , kfold);
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
        Model_WELM = WELM([2*Xtr_balance-1 , Ytr_balance]);
        Yts_predicted = elmpredict(Model_WELM,2*Xts-1);
        
        [Acc_WELM(kfold,iter) , GM_WELM(kfold,iter)] = GetEvaluationMetrics(Yts , Yts_predicted);
        
        if mod(iter,10) == 1
            disp(['WELM Accuracy is: ',num2str(Acc_WELM(kfold)), '%']);
            disp(['WELM GM is: ',num2str(GM_WELM(kfold)), '%']);
            disp('_________________________________');
        end
    end
end

disp(['WELM Average Accuracy is: ',num2str(mean(Acc_WELM(:))), '%']);
disp(['WELM Average STD Accuracy is: ',num2str(std(Acc_WELM(:)))]);

disp(['WELM Average GM is: ',num2str(mean(GM_WELM(:))), '%']);
disp(['WELM Average STD GM is: ',num2str(std(GM_WELM(:)))]);

save Model_WELM_balance Model_WELM
