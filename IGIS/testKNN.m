  function [Pc,PcVec,Result] = testKNN(R, Q, R_Class, Q_Class, K)
%
%  [Pc,PcVec,Result] = testKNN(R, Q, R_Class, Q_Class, K)
%  Songyot Nakariyakul 7-10-2016

%  fast KNN classifier , K is specified by user
%  Q is the query samples (Test): size of n samples x d feature numbers
%  R is the reference samples (Train): size of # m samples x d feature numbers
%  Q_Class is the n x 1 vector representing the class of the query
%  samples(1, 2, 3, etc) or (-1, 1)
%  R_Class is the m x 1 vector representing the class of the reference
%  samples (1, 2, 3, etc)  or (-1, 1)
%
%  Pc is the overall Pc score
%  PcVec is a vector containing the Pc score for each class
%  Result is an n x 1 vector containing the class result of each query sample predicted by KNN 


IDX = knnsearch(R,Q,'K',K);  % search for K nearest neighbor
Result = mode(R_Class(IDX),2); % classify K neighbors of each query sample and find the majority class

numMisclassify = nnz(Result~=Q_Class);
Pc = (1-numMisclassify/length(Q_Class))*100;

[C] = unique(R_Class);
nclass = length(C);         % find how many classes
PcVec = zeros(1,nclass);    % store the Pc score for each class
for ii = 1:nclass
    numClass = nnz(Q_Class == C(ii));
    PcVec(ii) = nnz(Result(Q_Class==C(ii))==C(ii))/numClass;
end

 