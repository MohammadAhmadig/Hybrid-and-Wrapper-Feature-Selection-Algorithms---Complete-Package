  function [Pc,PcVec,Result] = trainKNN(R, R_Class, K)
%
%  [Pc,PcVec,Result] = fastKNN(R, Q, R_Class, Q_Class, K)
%  Songyot Nakariyakul 1-21-2017

%  KNN classifier for training, K is specified by user
%  R is the reference samples (Train): size of # m samples x d feature numbers
%  R_Class is the m x 1 vector representing the class of the reference
%  samples (1, 2, 3, etc)  or (-1, 1)
%
%  Pc is the overall Pc score
%  PcVec is a vector containing the Pc score for each class
%  Result is an n x 1 vector containing the class result of each query sample predicted by KNN 


[IDX,~] = knnsearch(R,R,'K',K+1);
Result = mode(R_Class(IDX(:,2:K+1)),2); % classify K neighbors of each query sample and find the majority class

numMisclassify = nnz(Result~=R_Class);
Pc = (1-numMisclassify/length(R_Class))*100;

[C] = unique(R_Class);
nclass = length(C);         % find how many classes
PcVec = zeros(1,nclass);    % store the Pc score for each class
for ii = 1:nclass
    numClass = nnz(R_Class == C(ii));
    PcVec(ii) = nnz(Result(R_Class==C(ii))==C(ii))/numClass;
end

