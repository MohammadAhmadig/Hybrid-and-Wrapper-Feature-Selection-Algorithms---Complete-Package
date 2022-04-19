function [ Pc ] = classifier( train,test,groundtrain,groundtest,opt )
% Songyot Nakariyakul
% [ Pc, predicted ] = classifier( train,test,groundtrain,groundtest,mode )
% Pc is the accuracy rate (%), predicted is a vector of the predicted (class) output
%   opt = 1, k-NN classifier
%   opt = 2, Decision tree c45
%   opt = 3, NB classifier
%   opt = 4, SVM SMO

if opt == 1
    %Pc = ClassifyKnn_Test([train groundtrain],[test groundtest]);
    k = 1;      % 1-NN classifier
    if((length(train)~=length(test))||nnz(mean(train)~=mean(test)))  %knn test
        [Pc,~,predicted] = testKNN(train,test,groundtrain,groundtest,k);
    else   %knn train
        [Pc,~,predicted] = trainKNN(train,groundtrain,k);
    end
elseif opt == 2  % decision tree
    Pc = ClassifyC45_Test([train groundtrain],[test groundtest]);
%     tc = fitctree(train,groundtrain);
%     predicted = predict(tc,test);
%     Pc = nnz(predicted==groundtest)/length(groundtest)*100;
elseif opt == 3  % decision tree
    Pc = ClassifyNB_Test([train groundtrain],[test groundtest]);
elseif opt == 4  % decision tree
    Pc = ClassifySVM_Test([train groundtrain],[test groundtest]);
elseif opt == 5  % decision tree
    Pc = ClassifyRF_Test([train groundtrain],[test groundtest]);
else
    disp('Invalid classifier');
end

end

