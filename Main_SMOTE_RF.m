clc
clear
close all
load E2.mat
X = E2(: , 1:end-1);
Y = E2(: , end);

X = Normalize(X);
K1 = 10;
cvidx1 = crossvalind('Kfold' , size(X,1) , K1);
Acc_RF = zeros(K1,1);
GM_RF  = zeros(K1,1);

for kfold1 = 1 : K1
    
    [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx1 , kfold1);
    
    IncPer = 100*sum(Ytr==1)/sum(Ytr==2);
    XSMOTE = mySMOTE(Xtr(Ytr==2,:) , IncPer , 3);
    
    
    Xtr_balance = [Xtr(Ytr==1,:) ; XSMOTE];
    Ytr_balance = [ones(sum(Ytr==1),1) ; 2*ones(size(XSMOTE,1),1)];
    
    Xtr = Xtr_balance;
    Ytr = Ytr_balance;
    
    Model_RF = fitcensemble(Xtr_balance , Ytr_balance);
    Yts_predicted = predict(Model_RF , Xts );
    [Acc_RF(kfold1) , GM_RF(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    
   
    disp(['Random Forest Accuracy is: ',num2str(Acc_RF(kfold1)), '%']);
    disp(['Random Forest GM is: ',num2str(GM_RF(kfold1)), '%']);
    disp('_________________________________');
end


disp(['Random Forest Average Accuracy is: ',num2str(mean(Acc_RF)), '%']);
disp(['Random Forest Average STD Accuracy is: ',num2str(std(Acc_RF))]);

disp(['Random Forest Average GM is: ',num2str(mean(GM_RF)), '%']);
disp(['Random Forest Average STD GM is: ',num2str(std(GM_RF))]);
