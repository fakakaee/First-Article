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
Acc_KSVM = zeros(K1,1);
GM_KSVM  = zeros(K1,1);

Acc_LSVM = zeros(K1,1);
GM_LSVM  = zeros(K1,1);


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
    Model_KNN = fitcknn(Xtr , Ytr,'NumNeighbors',k);
    Ypot = predict(Model_KNN, Xpot);
    
    Xtr_balance = [Xtr ; Xpot(Ypot == 2 ,:)]; % to do: check if minority class be too large
    Ytr_balance = [Ytr ; Ypot(Ypot == 2)];
    
    Xtr = Xtr_balance;
    Ytr = Ytr_balance;
    
    K2 = 5;
    cvidx2 = crossvalind('Kfold' , size(Xtr,1) , K2);
    
    ParamRange = 10 .^ (-3: 3);
    GM_Eval_LSVM = zeros(K2 , length(ParamRange));
    GM_Eval_KSVM = zeros(K2 , length(ParamRange));
    for kfold2 = 1 : K2
        
        [Xtr_Nes, Ytr_Nes , Xev, Yev] = SplitTrainTest(Xtr , Ytr , cvidx2 , kfold2);
        
        for par = 1:length(ParamRange)
            
            Model_LSVM = fitcsvm(Xtr_Nes , Ytr_Nes,'BoxConstraint',ParamRange(par));
            Yev_predicted = predict(Model_LSVM , Xev );
            [~ , GM_Eval_LSVM(K2,par)] = GetEvaluationMetrics(Yev , Yev_predicted);
            
            Model_KSVM = fitcsvm(Xtr_Nes , Ytr_Nes,'BoxConstraint',ParamRange(par),'KernelFunction','gaussian','KernelScale',ParamRange(par));
            Yev_predicted = predict(Model_KSVM , Xev );
            [~ , GM_Eval_KSVM(K2,par)] = GetEvaluationMetrics(Yev , Yev_predicted);
            
        end
    end
    
    [~,Ipar] = max(mean(GM_Eval_LSVM));
    
    Model_LSVM = fitcsvm(Xtr_balance , Ytr_balance,'BoxConstraint',ParamRange(Ipar));
    Yts_predicted = predict(Model_LSVM , Xts );
    [Acc_LSVM(kfold1) , GM_LSVM(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    
    [~,Ipar] = max(mean(GM_Eval_KSVM));
    
    Model_KSVM = fitcsvm(Xtr_balance , Ytr_balance,'BoxConstraint',ParamRange(Ipar),'KernelFunction','gaussian','KernelScale',ParamRange(Ipar));
    Yts_predicted = predict(Model_KSVM , Xts );
    [Acc_KSVM(kfold1) , GM_KSVM(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    for kfold = 1 : K1
        n = 1;
        A(kfold,n) = Acc_KSVM(kfold);
        n = n+1 ;
        A(kfold, n) = GM_KSVM(kfold);
        n = n+1 ;
        A(kfold,n) = Acc_LSVM(kfold);
        n=n+1 ;
        A(kfold,n) = GM_LSVM(kfold);
        n=+1;
        
    end 
    filename = 'result.xlsx';
    xlswrite (filename,A);
    
    disp(['LSVM Accuracy is: ',num2str(Acc_LSVM(kfold1)), '%']);
    disp(['LSVM GM is: ',num2str(GM_LSVM(kfold1)), '%']);
   
    disp(['KSVM Accuracy is: ',num2str(Acc_KSVM(kfold1)), '%']);
    disp(['KSVM GM is: ',num2str(GM_KSVM(kfold1)), '%']);
    disp('_________________________________');
end

disp(['LSVM Average Accuracy is: ',num2str(mean(Acc_LSVM)), '%']);
disp(['LSVM Average STD Accuracy is: ',num2str(std(Acc_LSVM))]);

disp(['LSVM Average GM is: ',num2str(mean(GM_LSVM)), '%']);
disp(['LSVM Average STD GM is: ',num2str(std(GM_LSVM))]);


disp(['KSVM Average Accuracy is: ',num2str(mean(Acc_KSVM)), '%']);
disp(['KSVM Average STD Accuracy is: ',num2str(std(Acc_KSVM))]);

disp(['KSVM Average GM is: ',num2str(mean(GM_KSVM)), '%']);
disp(['KSVM Average STD GM is: ',num2str(std(GM_KSVM))]);



save Model_SVMs Model_LSVM Model_KSVM