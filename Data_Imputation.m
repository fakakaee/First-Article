clear
clc
[~,~,Data] = xlsread('clean data-v6-27april2019.xlsx');
Data(1,:) = [];
Data(end,:) = [];

T = Data(:,1);
T = cell2table(T);
T = table2array(T);
T = grp2idx(T);
Data(:,1) = num2cell(T);

T = Data(:,end);
T = cell2table(T);
T = table2array(T);
T = grp2idx(T);
Data(:,end) = num2cell(T);

Data = cell2mat(Data);

Y = Data(:, end);
X = knnimpute(Data(:,1:end-1)')';

save('Data_Imp_KNN.mat','X','Y');

X = fillmissing(Data(:,1:end-1) , 'linear');
save('Data_Imp_Linear.mat','X','Y');

X = fillmissing(Data(:,1:end-1) , 'spline');
save('Data_Imp_Spline.mat','X','Y');


X = fillmissing(Data(:,1:end-1) , 'pchip');
save('Data_Imp_Pchip.mat','X','Y');

