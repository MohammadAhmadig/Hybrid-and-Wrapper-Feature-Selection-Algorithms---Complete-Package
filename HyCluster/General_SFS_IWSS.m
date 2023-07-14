
clear;clc;
%## paths
WEKA_HOME = 'C:\Program Files\Weka-3-8';
javaaddpath('\weka.jar');
K = 10;% kFold
%Datasets = {'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat', 'pixraw10P.mat' };%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
Datasets = { 'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat', 'pixraw10P.mat',...
    'GLI-85.mat', 'CLL_SUB_111.mat' , 'SMK_CAN_187.mat','TOX_171.mat'};%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
%Datasets = {'warpPIE10P.mat'};%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
% %'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','orlraws10P.mat',
for DataName = Datasets
HOME = 'dataset_new\';
load([HOME, cell2mat(DataName)]);
if(strcmp(cell2mat(DataName),'GLI-85.mat') || strcmp(cell2mat(DataName),'CLL_SUB_111.mat') || ...
    strcmp(cell2mat(DataName),'SMK_CAN_187.mat') || strcmp(cell2mat(DataName),'TOX_171.mat'))
    X=zscore(X,1);%%%%%%%%%%%%%%%%%%%%%%%%% normalize
end 

OrgData =[X Y];
Accuracy = zeros(1,K);
NumberOfSelectedF = zeros(1,K);
time1 = zeros(1,K);

CVO = cvpartition(Y,'k',K); % Stratified cross-validation
for GroupSize = 1:1%(relevantSize/2)

for fold = 1:CVO.NumTestSets
    trIdx = CVO.training(fold);
    teIdx = CVO.test(fold);
    data = OrgData(trIdx,:);
    relevantSize = size(data,2)-1;
tic;
NodesSU =[];
for i = 1 : size(data,2)-1
    NodesSU(i) = (2*mi(data(:,i),data(:,end)))/(h(data(:,i)) + h(data(:,end)));
end
[~,sortedSUClass] = sort(NodesSU,'descend');% first is max

% remove irrelevant features 
TETA = 0.095;
%relevantSize = 100;%floor(sqrt(numAttrOriginal-1) * log(numAttrOriginal-1));
%irrelevantFeatures = sortedSUClass(relevantSize+1:end);
relevantFeatures = sortedSUClass(1:relevantSize);

y = data(:,end);
data = [data(:,relevantFeatures) data(:,end)];%%%%%%%%

% detect clusters
% Make IDX for random grouping
% IDX=[];
% for i=1:GroupSize
%     IDX = [IDX ; randperm(floor(relevantSize/GroupSize),floor(relevantSize/GroupSize))'];
% end
% clusters ={};
% IDX = [IDX;ceil(relevantSize/GroupSize)*ones(relevantSize-length(IDX),1)];
IDX = [1:relevantSize]';

numOfClusters = ceil(relevantSize/GroupSize);
features =[];clusterSizes =[];
for i = 1: numOfClusters
    [~,idx] = find(ismember(IDX(1:end)',i));% label feature akhare (dige nist)
    if(~isempty(idx))
     clusters{i} = idx;
     clusterSizes(i) = length(idx);
%     temp0 = NodesSU(idx);
%     [~,max0] = max(temp0);
%     %[~,sortedInerCluster{i}] = sort(temp0,'descend');
%     features(i) = idx(max0);% nemayandehaye har cluster
    end
end
OLDclusters=clusters;
% [~,sortedCluster] = sort(NodesSU(features),'descend');
numOfClusters = size(clusters,2);


selectedF =[];results = [];
cvo = cvpartition(y,'k',5);
for i = 1: numOfClusters
    index = i
    MaxIter = size(clusters{index},2);
    if(MaxIter > 40)
        MaxIter = floor(sqrt(size(clusters{index},2)));
    end
    newCluster = [selectedF clusters{index}];
    keepIn = [ones(1,length(selectedF)) zeros(1,length(clusters{index}))];
    keepIn = logical(keepIn);
    x = data(:,newCluster);%%%%%%%%
    
    [ fs ] = mySFS_IWSS( [x y] , keepIn ,'NB' ,cvo);
%     c = cvpartition(y,'k',5);
%     opts = statset('display','iter','MaxIter',MaxIter);
%     fun = @(xTrain,yTrain,xTest,yTest)(ClassifierC45(xTrain, yTrain, xTest, yTest));
%     [fs,history] = sequentialfs(fun,x,y,'cv',c,'options',opts,'keepin',keepIn);%%%%%
%     if(sum(fs) > length(selectedF))% if feature added with wrapper
%         results(1,i) = ClassifySVM([data(:,newCluster(find(fs))) y]);
%         results(2,i) = sum(fs);
%     end
    selectedF = newCluster(find(fs));
    
    selectedF
    %keepOut = selectedF;%%%
end

Accuracy(GroupSize,fold) = ClassifyNB_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
NumberOfSelectedF(GroupSize,fold) = length(selectedF);
time1(GroupSize,fold) = toc;
newDATA = data(:,selectedF);
newDATA = [newDATA y];

end

end
meanAcc=mean(Accuracy,2)
save(['results_NB_IWSS_fold_mySFS_minfold2\' cell2mat(DataName)],'CVO','meanAcc','Accuracy','NumberOfSelectedF','selectedF','time1');
end

