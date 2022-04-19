clear all;clc;

% Songyot Nakariyakul
% 8 Nov 2018
WEKA_HOME = 'C:\Program Files\Weka-3-8';
javaaddpath('\weka.jar');
% Reference:
% S. Nakariyakul, A hybrid gene selection algorithm based on interaction information for microarray-based cancer classification, PLOS One (2018).


%load COLON%pixraw10P
Datasets = { 'breast-cancer-wisconsin.mat','sonar.mat','wine.mat'};
%Datasets = {'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat', 'pixraw10P.mat' };%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
%Datasets = { 'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat', 'pixraw10P.mat',...
%Datasets = { 'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat', 'pixraw10P.mat',...
%    'GLI-85.mat', 'CLL_SUB_111.mat' ,'TOX_171.mat', 'SMK_CAN_187.mat'};%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
%Datasets = {'warpAR10P.mat'};%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
% %'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','orlraws10P.mat',
Classifiers = {'Knn','C45','NB','SVM','RF'};
for DataName = Datasets
HOME = 'datajadid\';
%HOME = 'dataset_new\';
load([HOME, cell2mat(DataName)]);
dataHOME = 'results2_VI_SU_fold\';
load([dataHOME, cell2mat(DataName)],'CVO');
class = Y; data = X;   

% microarray data: the first column is the class vector
%class = data(:,1); data = data(:,2:end);   

% data normalization
input=(data-repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));

for classifierID = 1:5
%opt = 1;            % select classifier: 1 = 3NN classifier, 2 = decision tree
K = 10;              % K = 5-fold cross-validation
selectf1 = [];      % selectf1 is the list of selected genes
%CVO=[];
time1 = [];meanACtest1 =[];meannumftr1 =[];meanJ1 =[];Accuracy=[];
for xx = 1:1           % 10 runs of 5-fold cross-validation 
    %CVO(xx) = cvpartition(class,'k',K); 
    numftr1 =[]; J1=[]; actest1=[];  temptime=[];
    for ii = 1:K       % run K-fold cross-validation
        ii
        tic;
        % create training set (K-1 partitions) and test set (1 partition)
        idxTrn = CVO(xx).training(ii); idxTst = CVO(xx).test(ii);
        Trn = input(idxTrn,:) ; Tst = input(idxTst,:);      % training set and test set
        LabelTrn = class(idxTrn); LabelTst = class(idxTst); % class labels for training and test sets
        
        [f1,acval1,actr1,numJ1] = IGIS(input, class, CVO(xx), ii, classifierID, K);  % run IGIS+
        numftr1(ii) = length(f1);       % number of selected genes
        J1(ii) = numJ1;                 % number of wrapper evaluations
        selectf1 = [selectf1; f1'];     % list of selected genes
        [ actest1(ii) ] = classifier(Trn(:,f1),Tst(:,f1),LabelTrn,LabelTst,classifierID); % compute test set accuracy
        temptime(ii) = toc;
        Accuracy(xx ,ii) = actest1(ii);
    end
    
    % compute the average of K-fold cross-validation of the xx-th run
    meanACtest1(xx) = mean(actest1);    % meanACtest1 = 10 averages of the K-fold CV test set accuracy 
    meannumftr1(xx) = mean(numftr1);    % meannumftr1 = 10 averages of the K-fold CV number of selected genes
    meanJ1(xx) = mean(J1);              % meanJ1 = 10 averages of the K-fold CV number of wrapper evaluations
    time1(xx) = mean(temptime);    
end

meanAcc=mean(meanACtest1)
NumberOfSelectedF = mean(meannumftr1)
numEval = mean(meanJ1)
save(['results2_',Classifiers{classifierID}, '_IGIS_fold\' cell2mat(DataName)],'CVO','meanAcc','Accuracy','NumberOfSelectedF','selectf1','time1','numEval');
end
end
