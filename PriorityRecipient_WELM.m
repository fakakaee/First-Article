clc
clear
close all
load E2.mat
rng(1);
X = E2(: , 1:end-1);
Y = E2(: , end);
Meld = X(:,3);
X = Normalize(X);
Donor_idx = [5 , 6 ,15];
Recipient_idx = setdiff(1 : size(X,2) , Donor_idx);


recIdx = randperm(size(X,1),20);
donIdx = randperm(size(X,1),10);

[~, idx] = sort(X(recIdx , 3));
recIdx = recIdx(idx);


% donIdx = 1 : 2 : size(X,1);



% [~ , recIdx] = sort(X(:,3),'descend');

% P = zeros(size(X,1),1);
L = zeros(20,10);

load Model_WELM.mat
for rec = 1 : 20
    x = zeros(1 , size(X,2));
    x(1 , Recipient_idx) = X(recIdx(rec) , Recipient_idx);
    
    for don = 1 : length(donIdx)
        x(1, Donor_idx) = X( donIdx(don) , Donor_idx);
        L(rec,don) = elmpredict(Model_WELM , x);
%         if l == 1
%             P(recIdx(rec)) = P(recIdx(rec)) + 1;
%         end
    end
%     P(recIdx(rec)) = P(recIdx(rec)) / length(donIdx);
    
end

% 
% [~ , IsortP] = sort(P, 'descend');
% IsortP(1:20)'

L(L==2) = 0;
Tabel = zeros(22,13);
Tabel(2:21 , 3:12) = L;
Tabel(1 , 3:12) = donIdx;
Tabel(2 : 21,1) = recIdx';
Tabel(2 : 21,2) = round(Meld(recIdx));
Tabel(end , 3:12) = sum(L);
Tabel(2:21 , end) = sum(L,2);
disp(Tabel)

