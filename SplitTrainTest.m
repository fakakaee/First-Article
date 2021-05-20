function [Xtr, Ytr , Xts, Yts] = SplitTrainTest(X , Y , cvidx , kfold)

Xts = X(cvidx == kfold , :);
Yts = Y(cvidx == kfold);

Xtr = X(cvidx ~= kfold , :);
Ytr = Y(cvidx ~= kfold );

end

