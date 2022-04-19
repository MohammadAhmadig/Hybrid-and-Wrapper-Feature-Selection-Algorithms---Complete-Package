% Written by Mohammad Ahmadi
clear;clc;
%## paths
WEKA_HOME = 'C:\Program Files\Weka-3-8';
javaaddpath('\weka.jar');
K = 10;% kFold

Datasets = { 'sonar.mat'};

for DataName = Datasets
    %'warpAR10P.mat', 'pixraw10P.mat',
HOME = 'data\';
load([HOME, cell2mat(DataName)]);
   
%X=zscore(X,1);%%%%%%%%%%%%%%%%%%%%%%%%% normalize
OrgData =[X Y];
AccuracyOrg = zeros(5,K);
Accuracy = zeros(5,K);
NumberOfSelectedF = zeros(5,K);
NumberOfSelectedFOrg = zeros(5,K);
avgSizeClusters = zeros(5,K);
time1 = zeros(5,K);
numOfEvals = zeros(5,K);

CVO = cvpartition(Y,'k',K); % Stratified cross-validation
for fold = 1:CVO.NumTestSets
    tic;
    trIdx = CVO.training(fold);
    teIdx = CVO.test(fold);
    data = OrgData(trIdx,:);
    
numAttrOriginal = size(data,2);

MaxIter =5;

NodesSU =[];
for i = 1 : size(data,2)-1
    NodesSU(i) = (2*mi(data(:,i),data(:,end)))/(h(data(:,i)) + h(data(:,end)));
    %NodesSU(i) = (2*(h(data(:,end)) - condh(data(:,end),data(:,i)) )/(h(data(:,i)) + h(data(:,end))) );
end
[~,sortedSUClass] = sort(NodesSU,'descend');% first is max

% remove irrelevant features 
TETA = 0.095;

% data(:,irrelevantFeatures) = [];
% NodesSU(:,irrelevantFeatures) = [];
NodesSU_dist = 1./NodesSU;


for classifierID = [1 2]
relevantSize = 30;%floor(sqrt(numAttrOriginal-1) * log(numAttrOriginal-1));
irrelevantFeatures = sortedSUClass(relevantSize+1:end);
relevantFeatures = sortedSUClass(1:relevantSize);
y = data(:,end);
x = data(:,relevantFeatures);%%%%%%%%
selectedF = [];

c = cvpartition(y,'k',5);
opts = statset('display','iter');
if(classifierID == 1)%strcmp(cell2mat(ClassifierName),'Knn')
    fun = @(xTrain,yTrain,xTest,yTest)(ClassifierKnn(xTrain, yTrain, xTest, yTest));
elseif(classifierID == 2)
    fun = @(xTrain,yTrain,xTest,yTest)(ClassifierC45(xTrain, yTrain, xTest, yTest));
elseif(classifierID == 3)
    fun = @(xTrain,yTrain,xTest,yTest)(ClassifierNB(xTrain, yTrain, xTest, yTest));
elseif(classifierID == 4)
    fun = @(xTrain,yTrain,xTest,yTest)(ClassifierSVM(xTrain, yTrain, xTest, yTest));
elseif(classifierID == 5)
    fun = @(xTrain,yTrain,xTest,yTest)(ClassifierRF(xTrain, yTrain, xTest, yTest));
end
[fs,history] = sequentialfs(fun,x,y,'cv',c,'options',opts);%%%%%

selectedF = relevantFeatures(find(fs));
BufferSize = 30;

while length(irrelevantFeatures)>BufferSize
    NumberOfSelecF = length(selectedF)
    CMI = zeros(size(irrelevantFeatures));
    for i = 1 : length(irrelevantFeatures)
        temp=[];
        for j = 1 : NumberOfSelecF
            temp(j) = cmi(data(:,irrelevantFeatures(i)),y,data(:,selectedF(j)));
            %2*mi(OrgDATA(:,irrelevantFeatures(i)),data(:,selectedF(j))))/(h(OrgDATA(:,irrelevantFeatures(i))) + h(data(:,selectedF(j))));
            
        end
        CMI(i) = min(temp);
    end
    
    [~,sortedCMI] = sort(CMI,'descend');
    
    newCluster = [selectedF irrelevantFeatures(sortedCMI(1:BufferSize))];
    x = data(:,newCluster);
    keepIn = [ones(1,length(selectedF)) zeros(1,BufferSize)];
    keepIn = logical(keepIn);
    c = cvpartition(y,'k',5);
    opts = statset('display','iter');
    if(classifierID == 1)%strcmp(cell2mat(ClassifierName),'Knn')
        fun = @(xTrain,yTrain,xTest,yTest)(ClassifierKnn(xTrain, yTrain, xTest, yTest));
    elseif(classifierID == 2)
        fun = @(xTrain,yTrain,xTest,yTest)(ClassifierC45(xTrain, yTrain, xTest, yTest));
    elseif(classifierID == 3)
        fun = @(xTrain,yTrain,xTest,yTest)(ClassifierNB(xTrain, yTrain, xTest, yTest));
    elseif(classifierID == 4)
        fun = @(xTrain,yTrain,xTest,yTest)(ClassifierSVM(xTrain, yTrain, xTest, yTest));
    elseif(classifierID == 5)
        fun = @(xTrain,yTrain,xTest,yTest)(ClassifierRF(xTrain, yTrain, xTest, yTest));
    end
    [fs,history] = sequentialfs(fun,x,y,'cv',c,'options',opts,'keepin',keepIn);%%%%%
    if(sum(fs) == length(selectedF)) %if not added feature with wrapper
        break;
    end
    selectedF = newCluster(find(fs));
%     x = x(:,find(fs));%%%%%%%%%%%%%%
%     Acc = Classify2([x , y]);
    %sizeOfNewSelectedF = size(x,2);
    irrelevantFeatures(sortedCMI(1:BufferSize))=[];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(classifierID == 1)%strcmp(cell2mat(ClassifierName),'Knn')
    Accuracy(classifierID,fold) = ClassifyKnn_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,selectedF) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = length(selectedF);
    %time1(classifierID,fold) = toc;
elseif(classifierID == 2)
    Accuracy(classifierID,fold) = ClassifyC45_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,selectedF) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = length(selectedF);
    time1(classifierID,fold) = toc;
elseif(classifierID == 3)
    Accuracy(classifierID,fold) = ClassifyNB_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,selectedF) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = length(selectedF);
    %time1(classifierID,fold) = toc;
elseif(classifierID == 4)
    Accuracy(classifierID,fold) = ClassifySVM_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,selectedF) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = length(selectedF);
    %time1(classifierID,fold) = toc;
elseif(classifierID == 5)
    Accuracy(classifierID,fold) = ClassifyRF_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,selectedF) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = length(selectedF);
    %time1(classifierID,fold) = toc;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Accuracy(fold) = ClassifyC45_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,selectedF) , Y(teIdx)]);
% %[Accuracy,indAcc]= max(results(1,:));
% %selectedF = selectedF(1:results(2,indAcc))
% NumberOfSelectedF(fold) = length(selectedF)
% time1(fold) = toc

%save(['results2\' cell2mat(DataName)],'Accuracy','optK','NumberOfSelectedF','avgSizeClusters','selectedF','newDATA','Acc','sizeOfNewSelectedF');
end
end
meanAcc = mean(Accuracy,2);
save(['result\' cell2mat(DataName)],'CVO','meanAcc','Accuracy','NumberOfSelectedF','selectedF','time1');% save in result folder
end

