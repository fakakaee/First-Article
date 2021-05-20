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
Acc_KNN = zeros(K1,1);
GM_KNN  = zeros(K1,1);

for kfold1 = 1 : K1
    
    [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx1 , kfold1);
    
    IncPer = 100*sum(Ytr==1)/sum(Ytr==2);
    XSMOTE = mySMOTE(Xtr(Ytr==2,:) , IncPer , 3);
    
    
    Xtr_balance = [Xtr(Ytr==1,:) ; XSMOTE];
    Ytr_balance = [ones(sum(Ytr==1),1) ; 2*ones(size(XSMOTE,1),1)];
    
    Xtr = Xtr_balance;
    Ytr = Ytr_balance;
    
    Model_KNN = fitcknn(Xtr_balance , Ytr_balance);
    Yts_predicted = predict(Model_KNN , Xts );
    [Acc_KNN(kfold1) , GM_KNN(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    
   
    disp(['KNN Accuracy is: ',num2str(Acc_KNN(kfold1)), '%']);
    disp(['KNN GM is: ',num2str(GM_KNN(kfold1)), '%']);
    disp('_________________________________');
        n = 10 ;
        A(kfold1,n) = Acc_KNN(kfold1);
        n = n+1 ;
        A(kfold1, n) =GM_KNN(kfold1);
    end

filename = 'result.xlsx';
xlswrite (filename,A);




disp(['KNN Average Accuracy is: ',num2str(mean(Acc_KNN)), '%']);
disp(['KNN Average STD Accuracy is: ',num2str(std(Acc_KNN))]);

disp(['KNN Average GM is: ',num2str(mean(GM_KNN)), '%']);
disp(['KNN Average STD GM is: ',num2str(std(GM_KNN))]);
