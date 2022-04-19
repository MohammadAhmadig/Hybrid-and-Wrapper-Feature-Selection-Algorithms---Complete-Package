% Written by Mohammad Ahmadi

clear;clc;
%## paths
WEKA_HOME = 'C:\Program Files\Weka-3-8';
javaaddpath('\weka.jar');
K = 10;% kFold
Datasets = { 'sonar.mat'};

Classifiers = {'Knn','C45','NB','SVM','RF'};
for DataName = Datasets
HOME = 'data\';
load([HOME, cell2mat(DataName)]);

OrgData =[X Y];
AccuracyOrg = zeros(5,K);
Accuracy = zeros(5,K);
NumberOfSelectedF = zeros(5,K);
NumberOfSelectedFOrg = zeros(5,K);
avgSizeClusters = zeros(5,K);
time1 = zeros(5,K);
numOfEvals = zeros(5,K);
GroupSize = 1;

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
relevantFeatures = sortedSUClass(1:relevantSize);

y = data(:,end);
data = [data(:,relevantFeatures) data(:,end)];%%%%%%%%

IDX = [1:relevantSize]';

numOfClusters = ceil(relevantSize/GroupSize);
features =[];clusterSizes =[];
for i = 1: numOfClusters
    [~,idx] = find(ismember(IDX(1:end)',i));
    if(~isempty(idx))
     clusters{i} = idx;
     clusterSizes(i) = length(idx);

    end
end
OLDclusters=clusters;
numOfClusters = size(clusters,2);

for classifierID = [1 2 4 5]
tic;
selectedF =[];results = [];
cvo = cvpartition(y,'k',5);
for i = 1: numOfClusters
    index = i
    MaxIter = size(clusters{index},2);
    newCluster = [selectedF clusters{index}];
    keepIn = [ones(1,length(selectedF)) zeros(1,length(clusters{index}))];
    keepIn = logical(keepIn);
    x = data(:,newCluster);%%%%%%%%
    
    [ fs ] = mySFS_IWSS( [x y] , keepIn ,Classifiers{classifierID} ,cvo);
%     c = cvpartition(y,'k',5);
%     opts = statset('display','iter','MaxIter',MaxIter);
%     fun = @(xTrain,yTrain,xTest,yTest)(ClassifierC45(xTrain, yTrain, xTest, yTest));
%     [fs,history] = sequentialfs(fun,x,y,'cv',c,'options',opts,'keepin',keepIn);%%%%%

    selectedF = newCluster(find(fs));
    
    selectedF
    %keepOut = selectedF;%%%
end

if(classifierID == 1)%strcmp(cell2mat(ClassifierName),'Knn')
    Accuracy(classifierID,fold) = ClassifyKnn_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    time1(classifierID,fold) = toc;
elseif(classifierID == 2)
    Accuracy(classifierID,fold) = ClassifyC45_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    time1(classifierID,fold) = toc;
elseif(classifierID == 3)
    Accuracy(classifierID,fold) = ClassifyNB_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    time1(classifierID,fold) = toc;
elseif(classifierID == 4)
    Accuracy(classifierID,fold) = ClassifySVM_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    time1(classifierID,fold) = toc;
elseif(classifierID == 5)
    Accuracy(classifierID,fold) = ClassifyRF_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    time1(classifierID,fold) = toc;
end


end
end
end
meanAcc=mean(Accuracy,2)
save(['result\' cell2mat(DataName)],'CVO','meanAcc','Accuracy','NumberOfSelectedF','selectedF','time1');%save in result folder
end

