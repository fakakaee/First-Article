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
    
    IncPer = 100*sum(Ytr==1)/sum(Ytr==2);
    XSMOTE = mySMOTE(Xtr(Ytr==2,:) , IncPer , 3);
    
    
    Xtr_balance = [Xtr(Ytr==1,:) ; XSMOTE];
    Ytr_balance = [ones(sum(Ytr==1),1) ; 2*ones(size(XSMOTE,1),1)];
    
    Xtr = Xtr_balance;
    Ytr = Ytr_balance;
    
    Model_NB = fitcnb(Xtr_balance , Ytr_balance);
    Yts_predicted = predict(Model_NB , Xts );
    [Acc_NB(kfold1) , GM_NB(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    
   
    disp(['Naive Bayes Accuracy is: ',num2str(Acc_NB(kfold1)), '%']);
    disp(['Naive Bayes GM is: ',num2str(GM_NB(kfold1)), '%']);
    disp('_________________________________');
    
        n = 9 ;
        A(kfold1,n) = Acc_NB(kfold1);
        n = n+1 ;
        A(kfold1, n) =GM_NB(kfold1);
    end

filename = 'resultNB.xlsx';
xlswrite (filename,A);   


disp(['Naive Bayes Average Accuracy is: ',num2str(mean(Acc_NB)), '%']);
disp(['Naive Bayes Average STD Accuracy is: ',num2str(std(Acc_NB))]);

disp(['Naive Bayes Average GM is: ',num2str(mean(GM_NB)), '%']);
disp(['Naive Bayes Average STD GM is: ',num2str(std(GM_NB))]);
