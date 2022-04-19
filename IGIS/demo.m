clear all

% Songyot Nakariyakul
% 8 Nov 2018

% Reference:
% S. Nakariyakul, A hybrid gene selection algorithm based on interaction information for microarray-based cancer classification, PLOS One (2018).


load COLON

% microarray data: the first column is the class vector
class = data(:,1); data = data(:,2:end);   

% data normalization
input=(data-repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));

opt = 1;            % select classifier: 1 = 3NN classifier, 2 = decision tree
K = 5;              % K = 5-fold cross-validation
selectf1 = [];      % selectf1 is the list of selected genes

for xx = 1:1           % 10 runs of 5-fold cross-validation 
    CVO = cvpartition(class,'k',K);      
    for ii = 1:K       % run K-fold cross-validation
        ii
        % create training set (K-1 partitions) and test set (1 partition)
        idxTrn = CVO(xx).training(ii); idxTst = CVO(xx).test(ii);
        Trn = input(idxTrn,:) ; Tst = input(idxTst,:);      % training set and test set
        LabelTrn = class(idxTrn); LabelTst = class(idxTst); % class labels for training and test sets
        
        [f1,acval1,actr1,numJ1] = IGISP(input, class, CVO(xx), ii, opt, K);  % run IGIS+
        numftr1(ii) = length(f1);       % number of selected genes
        J1(ii) = numJ1;                 % number of wrapper evaluations
        selectf1 = [selectf1; f1'];     % list of selected genes
        [ actest1(ii) ] = classifier(Trn(:,f1),Tst(:,f1),LabelTrn,LabelTst,opt); % compute test set accuracy
            
    end
    
    % compute the average of K-fold cross-validation of the xx-th run
    meanACtest1(xx) = mean(actest1);    % meanACtest1 = 10 averages of the K-fold CV test set accuracy 
    meannumftr1(xx) = mean(numftr1);    % meannumftr1 = 10 averages of the K-fold CV number of selected genes
    meanJ1(xx) = mean(J1);              % meanJ1 = 10 averages of the K-fold CV number of wrapper evaluations
        
end


