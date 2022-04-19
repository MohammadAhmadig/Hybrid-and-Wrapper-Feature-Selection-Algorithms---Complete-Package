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

for fold = 1:CVO.NumTestSets
    trIdx = CVO.training(fold);
    teIdx = CVO.test(fold);
    data = OrgData(trIdx,:);
    
numAttrOriginal = size(data,2);

NodesSU =[];
for i = 1 : size(data,2)-1
    NodesSU(i) = (2*mi(data(:,i),data(:,end)))/(h(data(:,i)) + h(data(:,end)));
end

[~,sortedSUClass] = sort(NodesSU,'descend');% first is max
% remove irrelevant features 
relevantSize = floor(sqrt(numAttrOriginal-1) * log(numAttrOriginal-1));

relevantFeatures = sortedSUClass(1:relevantSize);

y = data(:,end);
data = [data(:,relevantFeatures) data(:,end)];

for classifierID = 1:5
tic;

numOfEval = 0;
selectedF =[];results = [];
x = data(:,1:end-1);
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
% SFFS algorithm sequential floating forward selection
optsFirst = statset('display','iter','MaxIter',1);
ccounter = 0;
tempcounter = 0;
keepIn = zeros(1,relevantSize);
keepIn = logical(keepIn);
tempKeepIn = keepIn;
while(sum(ismember(keepIn,0)) > 0)
    flagg=0;
    [fs,history] = sequentialfs(fun,x,y,'cv',c,'options',optsFirst,'keepin',keepIn);%%%%%
    if( sum(keepIn == fs) == length(keepIn) )
        break;
    end
    if( ccounter == length(keepIn) || ccounter >= 100)
        break;
    end
    if(tempcounter >= 4 )
        flagg=1;
        tempcounter=0;
    end
    keepIn = fs;
    if(sum(keepIn == tempKeepIn)==length(keepIn))
        tempcounter = tempcounter +1;
    end
    tempKeepIn = keepIn;
    if( flagg == 0 )
        [fs,history] = sequentialfs(fun,x(:,keepIn),y,'cv',c,'options',opts,'direction','backward');
        keepIn(keepIn==1) = fs;
        fs = keepIn;
    end
    ccounter = ccounter+1;
    %     [ fs ] = mySFS( [x y] , keepIn ,'Knn' );
    
end

selectedF=fs;
    

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
meanAcc = mean(Accuracy,2)
%meanAccOrg = mean(AccuracyOrg)
%meanNumOfEval = mean(numOfEvals)
save(['result\'  cell2mat(DataName) ] ,'meanAcc','Accuracy','NumberOfSelectedF','selectedF','time1','CVO');% save in result folder

end

