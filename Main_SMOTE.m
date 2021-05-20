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
    
    IncPer = 100*sum(Ytr==1)/sum(Ytr==2);
    XSMOTE = mySMOTE(Xtr(Ytr==2,:) , IncPer , 3);
    
    
    Xtr_balance = [Xtr(Ytr==1,:) ; XSMOTE];
    Ytr_balance = [ones(sum(Ytr==1),1) ; 2*ones(size(XSMOTE,1),1)];
    
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
    
    
    Model_KSVM = fitcsvm(Xtr_balance , Ytr_balance,'KernelFunction','gaussian','KernelScale',1);
    Yts_predicted = predict(Model_KSVM , Xts );
    [Acc_KSVM(kfold1) , GM_KSVM(kfold1)] = GetEvaluationMetrics(Yts , Yts_predicted);
    
    disp(['LSVM Accuracy is: ',num2str(Acc_LSVM(kfold1)), '%']);
    disp(['LSVM GM is: ',num2str(GM_LSVM(kfold1)), '%']);
   
    disp(['KSVM Accuracy is: ',num2str(Acc_KSVM(kfold1)), '%']);
    disp(['KSVM GM is: ',num2str(GM_KSVM(kfold1)), '%']);
    disp('_________________________________');
    
end  
 for kfold1 = 1 : K1
        n = 10;
        A(kfold1,n) = Acc_KSVM(kfold1);
        n = n+1 ;
        A(kfold1, n) = GM_KSVM(kfold1);
        n = n+1 ;
        A(kfold1,n) = Acc_LSVM(kfold1);
        n=n+1 ;
        A(kfold1,n) = GM_LSVM(kfold1);
        n=+1;
        
    end 
    filename = 'result-16May.xlsx';
    xlswrite (filename,A);



disp(['LSVM Average Accuracy is: ',num2str(mean(Acc_LSVM)), '%']);
disp(['LSVM Average STD Accuracy is: ',num2str(std(Acc_LSVM))]);

disp(['LSVM Average GM is: ',num2str(mean(GM_LSVM)), '%']);
disp(['LSVM Average STD GM is: ',num2str(std(GM_LSVM))]);


disp(['KSVM Average Accuracy is: ',num2str(mean(Acc_KSVM)), '%']);
disp(['KSVM Average STD Accuracy is: ',num2str(std(Acc_KSVM))]);

disp(['KSVM Average GM is: ',num2str(mean(GM_KSVM)), '%']);
disp(['KSVM Average STD GM is: ',num2str(std(GM_KSVM))]);
