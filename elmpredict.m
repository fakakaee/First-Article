function L = elmpredict(Model,X)
X = X';
NumberofTestingData = size(X,2);

% 
% temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
% for i = 1:NumberofTestingData
%     for j = 1:number_class
%         if label(1,j) == TV.T(1,i)
%             break;
%         end
%     end
%     temp_TV_T(j,i)=1;
% end
% TV.T=temp_TV_T*2-1;

tempH_test=Model.InputWeight*X;
ind=ones(1,NumberofTestingData);
BiasMatrix=Model.BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(Model.ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);
        %%%%%%%% More activation functions can be added here
end
TY=(H_test' * Model.OutputWeight)';                       %   TY: the actual output of the testing data

%%%%%%%%%% Calculate training & testing classification accuracy
L = zeros(size(X,2),1);
for i = 1 : size(X,2)
    [~, L(i)] = max(TY(:,i));
end


end

