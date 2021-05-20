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
cvidx1 = crossvalind('Kfold' , size(X,1) , K1);
Acc_DT = zeros(K1,1);
GM_DT  = zeros(K1,1);

R = 200;
X = GoToFeatureSpace_RFF(X,R);


for kfold1 = 1 : K1
    
    [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx1 , kfold1);
    
    IncPer = 100*sum(Ytr==1)/sum(Ytr==2);
    XSMOTE = mySMOTE(Xtr(Ytr==2,:) , IncPer , 3);
    
    
    Xtr_balance = [Xtr(Ytr==1,:) ; XSMOTE];
    Ytr_balance = [ones(sum(Ytr==1),1) ; 2*ones(size(XSMOTE,1),1)];
    
    Xtr = Xtr_balance;
    Ytr = Ytr_balance;
    
    Model_DT = fitctree(Xtr_balance , Ytr_balance);
    Yts_predicted = predict(Model_DT , Xts );
    [Acc_DT(kfold1) , GM_DT(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    
   
    disp(['Tree Accuracy is: ',num2str(Acc_DT(kfold1)), '%']);
    disp(['Tree GM is: ',num2str(GM_DT(kfold1)), '%']);
    disp('_________________________________');
        n = 17 ;
        A(kfold1,n) = Acc_DT(kfold1);
        n = n+1 ;
        A(kfold1, n) = GM_DT(kfold1);
    end

filename = 'RWELM.xlsx';
xlswrite (filename,A);




disp(['Tree Average Accuracy is: ',num2str(mean(Acc_DT)), '%']);
disp(['Tree Average STD Accuracy is: ',num2str(std(Acc_DT))]);

disp(['Tree Average GM is: ',num2str(mean(GM_DT)), '%']);
disp(['Tree Average STD GM is: ',num2str(std(GM_DT))]);
