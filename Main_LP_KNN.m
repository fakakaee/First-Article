clc
clear
close all
load Data_Imp_KNN.mat
% X = E2(: , 1:end-1);
% Y = E2(: , end);
X = Normalize(X);
K1 = 10;

A = zeros (100,2);
 n = 0 ;

cvidx1 = crossvalind('Kfold' , size(X,1) , K1);
Acc_KNN = zeros(K1,1);
GM_KNN  = zeros(K1,1);

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
    
    
    
    Model_KNN = fitcknn(Xtr_balance , Ytr_balance);
    Yts_predicted = predict(Model_KNN , Xts );
    [Acc_KNN(kfold1) , GM_KNN(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    
   
    disp(['KNN Accuracy is: ',num2str(Acc_KNN(kfold1)), '%']);
    disp(['KNN GM is: ',num2str(GM_KNN(kfold1)), '%']);
    disp('_________________________________');
    
            n = 5 ;
        A(kfold1,n) = Acc_KNN(kfold1);
        n = n+1 ;
        A(kfold1, n) =GM_KNN(kfold1);
    end

filename = 'RWELM.xlsx';
xlswrite (filename,A);




disp(['KNN Average Accuracy is: ',num2str(mean(Acc_KNN)), '%']);
disp(['KNN Average STD Accuracy is: ',num2str(std(Acc_KNN))]);

disp(['KNN Average GM is: ',num2str(mean(GM_KNN)), '%']);
disp(['KNN Average STD GM is: ',num2str(std(GM_KNN))]);



save Model_LP_KNN Model_KNN