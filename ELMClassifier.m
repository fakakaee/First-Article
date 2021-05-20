function [Model,Y] = ELMClassifier(TrainingData_File)

% 
% for i=1:numel(lb)
%     TrainingData_File(:,i)=((TrainingData_File(:,i)-lb(i))./(ub(i)-lb(i)))*2-1;
%     TestingData_File(:,i)=((TestingData_File(:,i)-lb(i))./(ub(i)-lb(i)))*2-1;
% end


NumberofHiddenNeurons=170;
ActivationFunction='radbas';
%%%%%%%%%%% Load training dataset
train_data=[ TrainingData_File(:,end) TrainingData_File(:,1:end-1)];
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset


NumberofTrainingData=size(P,2);
% NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%%% Preprocessing the data of classification
% sorted_target=sort(cat(2,T,TV.T),2);
% label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
% label(1,1)=sorted_target(1,1);
% j=1;
% for i = 2:(NumberofTrainingData+NumberofTestingData)
%     if sorted_target(1,i) ~= label(1,j)
%         j=j+1;
%         label(1,j) = sorted_target(1,i);
%     end
% end

label(1,:) = unique(T);

number_class=length(label(1,:));
NumberofOutputNeurons=number_class;

%%%%%%%%%% Processing the targets of training
temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
for i = 1:NumberofTrainingData
    for j = 1:number_class
        if label(1,j) == T(1,i)
            break;
        end
    end
    temp_T(j,i)=1;
end
T=temp_T*2-1;

% %%%%%%%%%% Processing the targets of testing
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

%%%%%%%%%%% Calculate weights & biases

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
clear H;

%%%%%%%%%%% Calculate the output of testing input
Model.InputWeight = InputWeight;
Model.BiasofHiddenNeurons = BiasofHiddenNeurons;
Model.ActivationFunction = ActivationFunction;
Model.OutputWeight = OutputWeight;

end


