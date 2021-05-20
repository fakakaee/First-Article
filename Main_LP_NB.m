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


cvidx1 = crossvalind('Kfold' , size(X,1) , K1);
Acc_NB = zeros(K1,1);
GM_NB  = zeros(K1,1);

for kfold1 = 1 : K1
    
    [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx1 , kfold1);
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
    Model_KNN_LP = fitcknn(Xtr , Ytr,'NumNeighbors',k);
    Ypot = predict(Model_KNN_LP, Xpot);
    
    Xtr_balance = [Xtr ; Xpot(Ypot == 2 ,:)]; % to do: check if minority class be too large
    Ytr_balance = [Ytr ; Ypot(Ypot == 2)];
    
    Xtr = Xtr_balance;
    Ytr = Ytr_balance;
    
    
    
    Model_NB = fitcnb(Xtr_balance , Ytr_balance);
    Yts_predicted = predict(Model_NB , Xts );
    [Acc_NB(kfold1) , GM_NB(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    
   
    disp(['NB Accuracy is: ',num2str(Acc_NB(kfold1)), '%']);
    disp(['NB GM is: ',num2str(GM_NB(kfold1)), '%']);
    disp('_________________________________');
        n = 3 ;
        A(kfold1,n) = Acc_NB(kfold1);
        n = n+1 ;
        A(kfold1, n) =GM_NB(kfold1);
    end

filename = 'resultNB.xlsx';
xlswrite (filename,A);    



disp(['NB Average Accuracy is: ',num2str(mean(Acc_NB)), '%']);
disp(['NB Average STD Accuracy is: ',num2str(std(Acc_NB))]);

disp(['NB Average GM is: ',num2str(mean(GM_NB)), '%']);
disp(['NB Average STD GM is: ',num2str(std(GM_NB))]);



save Model_LP_NB Model_NB