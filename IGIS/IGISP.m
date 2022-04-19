function [featIndex, meanACCval, meanACCtr, numJ] = IGISP(inp, ground, CVO, i, opt, K)
% Songyot Nakariyakul
% improved interaction-guided incremental selection (IGIS+)

% Inputs:
%   inp has the size of numSamples x numFeatures
%   ground is the groundtruth of size numberSamples x 1
%   CVO is the cross-validation partition for data
%   i is the i-th run of K-fold outer cross-validation
%   opt is the classifier: 1 = 3NN classifier, 2 = decision tree
%   K is K-fold outer cross-validation

% Outputs:
%   featIndex is the lists of features selected
%   meanACCval is the average validation set accuracy rates
%   meanACCtr is the average training set accuracy rates
%   numJ is the number of wrapper evaluations

k = K-1;   % k = (K-1)fold cross-validation for the inner loop
fold = 1:(k+1);  fold(i) = [];

[~, numFtr] = size(inp);

ACval = zeros(1,k);  % ACval is the validation set accuracy rate
ACtr = zeros(1,k);   % ACtr is the training set accuracy rate

% Generate inner-loop index (1 or 0) for cross-validation. It has the size of numSamples x k
idxval = false(length(ground),k);  idxtrn = idxval;
for jj = 1:k
    idxval(:,jj) = CVO.test(fold(jj));
    xx = fold; xx(jj) = [];
    t = zeros(length(ground),1);
    for cc = xx
        t = t + CVO.test(cc);
    end
    idxtrn(:,jj) = logical(t);
end


ttime = cputime;  numJ = 0;

% Select the first feature which has the highest Pc score
idxTrn1 = CVO.training(i); 
input = inp(idxTrn1,:); groundtruth = ground(idxTrn1);
Pc = zeros(1,numFtr);
for ii = 1:numFtr
    Pc(ii) = classifier(input(:,ii),input(:,ii),groundtruth,groundtruth,opt);
    numJ = numJ + 1;
end

[~,ix] = sort(Pc);
f = fliplr(ix);
featIndex = f(1);       % f(1) is the first selected feature
ff = 1;

% compute the k-fold accuracy rates of training and validation sets using the first feature
for jj = 1:k
    
    Trn = inp(idxtrn(:,jj),:); LabelTrn = ground(idxtrn(:,jj));
    Tst = inp(idxval(:,jj),:); LabelTst = ground(idxval(:,jj));
    
    ACtr(1,jj) = classifier(Trn(:,featIndex),Trn(:,featIndex),LabelTrn,LabelTrn,opt);  numJ = numJ + 1;
    ACval(1,jj) = classifier(Trn(:,featIndex),Tst(:,featIndex),LabelTrn,LabelTst,opt); numJ = numJ + 1;
    
end

ind = 1;
maxACCtr = ACtr; 
maxACCval = ACval;
meanACCtr(ind) = mean(ACtr);
meanACCval(ind) = mean(ACval); 



[numTrn, numFtr] = size(input);
% Discretize the input before computing Shannon's Information Theory functions
tmp1 = input > repmat(mean(input)+std(input),numTrn,1);
tmp2 = input < repmat(mean(input)-std(input),numTrn,1);
TrnD = tmp1 - tmp2;

restFtrs = 1:numFtr;    % restFtrs is the list of available features

D = zeros(1,numFtr);   % D is the mutual information for numFtr features
for ii = 1:numFtr
    D(ii) = mi(TrnD(:,ii),groundtruth); % mutual information
end

restFtrs(f(1)) = 0;     % remove the first selected feature from the feature list
D(f(1)) = -500;         % remove the fist selected feature from D

W = zeros(numFtr,20);   % W stores the interaction information
q = 0;                  % q = 1, quit program


while isempty(find(restFtrs, 1)) == 0
    
    ftr = find(restFtrs);
    curr = featIndex(ind);
    % compute interaction information to find candidate features
    for ii = ftr
        W(ii,ind) = cmi(TrnD(:,ii),TrnD(:,curr),groundtruth)-mi(TrnD(:,ii),TrnD(:,curr));
    end
    
    w_mi = mean(W(:,1:ind),2);
    [jmi, idx] = sort(D + w_mi');
    jmi = fliplr(jmi);  idx = fliplr(idx);
    idx = idx(jmi > 0);
 
    lengthidx = length(idx);
    
    if lengthidx == 0,  break,   end
    
    for ss = 1:lengthidx
        %ss
        tmpFtr = [featIndex idx(ss)];
        
        sig = 0;
        ff = ff + 1;
         
        for jj = 1:k
            
            Trn = inp(idxtrn(:,jj),:); LabelTrn = ground(idxtrn(:,jj));
            
            ACtr(1,jj) = classifier(Trn(:,tmpFtr),Trn(:,tmpFtr),LabelTrn,LabelTrn,opt);
            numJ = numJ + 1;
            
        end
        
        if (cohen_d(ACtr, maxACCtr) >= 0.4)    % Use Cohen's d to test significant improvement for training set accuracy
                    
            % compute validation set accuracy
            for jj = 1:k
                Trn = inp(idxtrn(:,jj),:); LabelTrn = ground(idxtrn(:,jj));
                Val = inp(idxval(:,jj),:); LabelVal = ground(idxval(:,jj));
                ACval(1,jj) = classifier(Trn(:,tmpFtr),Val(:,tmpFtr),LabelTrn,LabelVal,opt);
                numJ = numJ + 1;
            end
            
            if (cohen_d(ACval, maxACCval) >= 0.15) % Use Cohen's d to test significant improvement for validation set accuracy                               
                % update the performance results
                sig = 1;
                ind = ind + 1;
                featIndex = tmpFtr;
                maxACCtr = ACtr;
                maxACCval = ACval;
                meanACCtr(ind) = mean(ACtr);
                meanACCval(ind) = mean(ACval);
            end
            
        end
                
        restFtrs(idx(ss)) = 0;  % remove the current feature from the feature list
        D(idx(ss)) = -500;      % remove the current feature from D
        
        if (mean(maxACCtr) == 100) || (mean(maxACCval) == 100)    % stopping criterion
            q = 1;
        end
        
        if sig == 1, break,  end     
        
    end
         
    if q == 1,  break,  end
     
end

fprintf('IGISP: numJ = %d, time = %d\n', numJ, round(cputime-ttime));

end

% calculate Cohen's d effect size
function [effsize] = cohen_d(x_after, x_before)

nt = length(x_after);
nc = length(x_before);
s_pool = sqrt(((nt-1)*var(x_after)+(nc-1)*var(x_before))/(nt+nc));
effsize = (mean(x_after)-mean(x_before))/s_pool;

end

