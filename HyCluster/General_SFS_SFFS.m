
clear;clc;
%## paths
WEKA_HOME = 'C:\Program Files\Weka-3-8';
javaaddpath('\weka.jar');
K = 10;% kFold
%Datasets = {'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat', 'pixraw10P.mat' };%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
Datasets = { 'warpAR10P.mat','warpPIE10P.mat', 'pixraw10P.mat',...
    'GLI-85.mat', 'CLL_SUB_111.mat' , 'SMK_CAN_187.mat','TOX_171.mat'};%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
% %'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','orlraws10P.mat',
Classifiers = {'Knn','C45','NB','SVM','RF'};
for DataName = Datasets
HOME = 'dataset_new\';
load([HOME, cell2mat(DataName)]);
dataHOME = 'results_Knn_Kmeans_IWSS_myMethod_fold_mySFS_minfold2\';
load([dataHOME, cell2mat(DataName)],'CVO');

if(strcmp(cell2mat(DataName),'GLI-85.mat') || strcmp(cell2mat(DataName),'CLL_SUB_111.mat') || ...
    strcmp(cell2mat(DataName),'SMK_CAN_187.mat') || strcmp(cell2mat(DataName),'TOX_171.mat'))
    X=zscore(X,1);%%%%%%%%%%%%%%%%%%%%%%%%% normalize
end 

OrgData =[X Y];
AccuracyOrg = zeros(5,K);
Accuracy = zeros(5,K);
NumberOfSelectedF = zeros(5,K);
NumberOfSelectedFOrg = zeros(5,K);
avgSizeClusters = zeros(5,K);
time1 = zeros(5,K);
numOfEvals = zeros(5,K);
GroupSize = 1;

%CVO = cvpartition(Y,'k',K); % Stratified cross-validation
%for GroupSize = 1:1%(relevantSize/2)

for fold = 1:CVO.NumTestSets
    trIdx = CVO.training(fold);
    teIdx = CVO.test(fold);
    data = OrgData(trIdx,:);
    
numAttrOriginal = size(data,2);
%tic;
%%% importing of classes
maxK =50;

if numAttrOriginal<100
    maxK =numAttrOriginal-1;
end

NodesSU =[];
for i = 1 : size(data,2)-1
    NodesSU(i) = (2*mi(data(:,i),data(:,end)))/(h(data(:,i)) + h(data(:,end)));
    %NodesSU(i) = (2*(h(data(:,end)) - condh(data(:,end),data(:,i)) )/(h(data(:,i)) + h(data(:,end))) );
end
%[~,sortedSUClass] = sort(NodesSU);% first is min
[~,sortedSUClass] = sort(NodesSU,'descend');% first is max
% remove irrelevant features 
TETA = 0.095;
relevantSize = floor(sqrt(numAttrOriginal-1) * log(numAttrOriginal-1));
%relevantSize = floor((numAttrOriginal-1)/2);

%irrelevantFeatures = sortedSUClass(relevantSize+1:end);
relevantFeatures = sortedSUClass(1:relevantSize);
%relevantSize = numAttrOriginal-1;
%relevantSize = floor((numAttrOriginal-1)/2);

%irrelevantFeatures = sortedSUClass(1:(numAttrOriginal-1-relevantSize));
% data(:,irrelevantFeatures) = [];
% NodesSU(:,irrelevantFeatures) = [];

y = data(:,end);
data = [data(:,relevantFeatures) data(:,end)];
%%%%%%%%%%% KMEANS %%%%%%%%%%
% [IDX,~,~,optK]=kmeans_opt(data(:,1:end-1)' , maxK);
% IDX;
%%%%%%%%%%% KMEANS %%%%%%%%%%

%%%%%%%%%%% SNN %%%%%%%%%%
% if(strcmp(cell2mat(DataName),'TOX_171.mat'))
%     KK=12;
%     minPoints=8;%round(0.7*KK);
%     ep=5;%round(0.5*KK);
% elseif(strcmp(cell2mat(DataName),'GLI-85.mat') || strcmp(cell2mat(DataName),'CLL_SUB_111.mat') )
%     KK=12;
%     minPoints=8;%round(0.7*KK);
%     ep=6;%round(0.5*KK);
% elseif(strcmp(cell2mat(DataName),'orlraws10P.mat') || strcmp(cell2mat(DataName),'warpPIE10P.mat'))
%     KK=14;
%     minPoints=8;%round(0.7*KK);
%     ep=8;%round(0.5*KK);
% elseif(strcmp(cell2mat(DataName),'warpAR10P.mat') || strcmp(cell2mat(DataName),'pixraw10P.mat') ||...
%     strcmp(cell2mat(DataName),'SMK_CAN_187.mat') )
%     KK=12;
%     minPoints=8;%round(0.7*KK);
%     ep=8;%round(0.5*KK);
% end
% [IDX,optK]=SNN2(data(:,1:end-1)',KK,ep,minPoints);
% uni=unique(IDX);
% optK=optK+1;
% IDX2=zeros(size(IDX));
% for i = 2: optK
%     IDX2(ismember(IDX,uni(i)))=i-1;
% end
% IDX2(ismember(IDX,0))=optK:(sum(ismember(IDX,0))+optK-1);
% optK=length(unique(IDX2));
% IDX=IDX2;
%%%%%%%%%%% SNN %%%%%%%%%%

% % calculate mutual su between all features
% numattr = size(data,2)
% Adjacency =[];
% for i = 1 : numattr-1
%     for j = 1 : numattr-1
%         if(j > i) % bara ye bar hesab kardan va kaheshe mohasebat
%             Adjacency(i,j) = (2*mi(data(:,i),data(:,j)))/(h(data(:,i)) + h(data(:,j)));
%         end
% %         if(i ~= j)
% %             Adjacency(i,j) = (2*mi(data(:,i),data(:,j)))/(h(data(:,i)) + h(data(:,j)));
% %         end
%     end
% end
% Adjacency=triu([Adjacency ; zeros(1,size(Adjacency,2))])+triu([Adjacency ; zeros(1,size(Adjacency,2))],1)';
% % Adjacency_dist = 1./Adjacency;%%%%%%%%%%%%%%%%%%%
% % 
% % %%%%%%%%%%% Spectral %%%%%%%%%%
% numOfEig = 6;
% [IDX,~,~,optK]=SpectralClustering(Adjacency ,numOfEig ,maxK ,3);
%%%%%%%%%%% Spectral %%%%%%%%%%

% detect clusters
% Make IDX for random grouping
% IDX=[];
% for i=1:GroupSize
%     IDX = [IDX ; randperm(floor(relevantSize/GroupSize),floor(relevantSize/GroupSize))'];
% end
% clusters ={};
% IDX = [IDX;ceil(relevantSize/GroupSize)*ones(relevantSize-length(IDX),1)];

% detect clusters
% clusters ={};
% sortedInerCluster ={};
% features =[];clusterSizes =[];
% for i = 1: optK
%     [~,idx] = find(ismember(IDX(1:end)',i));% label feature akhare (dige nist)
%     if(~isempty(idx))
%     clusters{i} = idx;
%     clusterSizes(i) = length(idx);
%     temp0 = NodesSU(idx);
%     [~,max0] = max(temp0);
%     [~,sortedInerCluster{i}] = sort(temp0,'descend');
%     features(i) = idx(max0);% nemayandehaye har cluster
%     end
% end
% OLDclusters=clusters;
% [~,sortedCluster] = sort(NodesSU(features),'descend');
% numOfClusters = size(clusters,2);

for classifierID = 4:5
tic;
% moshkel darad hanooz 
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
    
    %     if(sum(fs) > length(selectedF))% if feature added with wrapper
    %         results(1,i) = ClassifyKnn([data(:,newCluster(find(fs))) y]);
    %         results(2,i) = sum(fs);
    %     end
end

selectedF=fs
    

%%%%%%%%% edit for changing cluster style %%%%%%%%
% selectedF =[];results = [];
% for i = 1: size(clusters,2)
%     %index = sortedCluster(i);% strat from best cluster to worst cluster
%     MaxIter = size(clusters{i},2);
%     if(MaxIter > 40)
%         MaxIter = floor(sqrt(size(clusters{i},2)));
%     end
%     newCluster = [selectedF clusters{i}];%each cluster strat from best feature to worst feature
%     keepIn = [ones(1,length(selectedF)) zeros(1,length(clusters{i}))];
%     keepIn = logical(keepIn);
%     x = data(:,newCluster);%%%%%%%%
%     c = cvpartition(y,'k',5); 
%     opts = statset('display','iter','MaxIter',MaxIter);
%     fun = @(xTrain,yTrain,xTest,yTest)(ClassifierC45(xTrain, yTrain, xTest, yTest));
%     [fs,history] = sequentialfs(fun,x,y,'cv',c,'options',opts,'keepin',keepIn);%%%%%
%     %in karo kon: agar feature added with wrapper resultesh kamtar shod khob bargard o
%     % in feature jadidaro entekhab nakon
%     if(sum(fs) > length(selectedF))% if feature added with wrapper
%         results(1,i) = ClassifyC45([data(:,newCluster(find(fs))) y]);
%         results(2,i) = sum(fs);
%     end
%     selectedF = newCluster(find(fs));
%     
%     selectedF
%     %keepOut = selectedF;%%%
% end
%%%%%%%%% edit for changing cluster style %%%%%%%%

if(classifierID == 1)%strcmp(cell2mat(ClassifierName),'Knn')
    Accuracy(classifierID,fold) = ClassifyKnn_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    %avgSizeClusters(classifierID,fold) = mean(clusterSizes);
    time1(classifierID,fold) = toc;
elseif(classifierID == 2)
    Accuracy(classifierID,fold) = ClassifyC45_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    %avgSizeClusters(classifierID,fold) = mean(clusterSizes);
    time1(classifierID,fold) = toc;
elseif(classifierID == 3)
    Accuracy(classifierID,fold) = ClassifyNB_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    %avgSizeClusters(classifierID,fold) = mean(clusterSizes);
    time1(classifierID,fold) = toc;
elseif(classifierID == 4)
    Accuracy(classifierID,fold) = ClassifySVM_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    %avgSizeClusters(classifierID,fold) = mean(clusterSizes);
    time1(classifierID,fold) = toc;
elseif(classifierID == 5)
    Accuracy(classifierID,fold) = ClassifyRF_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
    NumberOfSelectedF(classifierID,fold) = sum(selectedF);
    %avgSizeClusters(classifierID,fold) = mean(clusterSizes);
    time1(classifierID,fold) = toc;
end
% %[~,indAcc]= max(results(1,:));
% selectedFOrg = selectedF;
% AccuracyOrg(fold) = ClassifyC45_Test([data(:,selectedFOrg) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedFOrg)) , Y(teIdx)])
% % selectedF = selectedF(1:results(2,indAcc));
% % Accuracy(fold) = ClassifyKnn_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
% % NumberOfSelectedF(fold) = length(selectedF)
% NumberOfSelectedFOrg(fold) = length(selectedFOrg)
% avgSizeClusters(fold) = mean(clusterSizes)
% time1(fold) = toc
% numOfEvals(fold) = numOfEval
% % newDATA = data(:,selectedF);
% % newDATA = [newDATA y];

%%%%%%%%%%%%% BUFFER CMI %%%%%%%%%%%%%%
% BufferSize = 30;
% CMI = zeros(size(irrelevantFeatures));
% for i = 1 : length(irrelevantFeatures)
%     temp=[];
%     for j = 1 : NumberOfSelectedF
%         temp(j) = cmi(Orginal_DATA(:,irrelevantFeatures(i)),y,data(:,selectedF(j)));
%         %2*mi(OrgDATA(:,irrelevantFeatures(i)),data(:,selectedF(j))))/(h(OrgDATA(:,irrelevantFeatures(i))) + h(data(:,selectedF(j))));
%         
%     end
%     CMI(i) = min(temp);
% end
% [~,sortedCMI] = sort(CMI,'descend');
% %MaxIter = floor(sqrt(BufferSize));
% x = [data(:,selectedF) OrgDATA(:,irrelevantFeatures(sortedCMI(1:BufferSize)))];
% keepIn = [ones(1,length(selectedF)) zeros(1,BufferSize)];
% keepIn = logical(keepIn);
% c = cvpartition(y,'k',10); 
% opts = statset('display','iter');
% fun = @(xTrain,yTrain,xTest,yTest)(Classifier2(xTrain, yTrain, xTest, yTest));
% [fs,history] = sequentialfs(fun,x,y,'cv',c,'options',opts,'keepin',keepIn);%%%%%
% x = x(:,find(fs));%%%%%%%%%%%%%%
% Acc = Classify2([x , y]);
% sizeOfNewSelectedF = size(x,2);
% save(['results2\' cell2mat(DataName)],'Accuracy','optK','NumberOfSelectedF','avgSizeClusters','selectedF','newDATA','Acc','sizeOfNewSelectedF');

%save(['results_NB_SNN2_newStyle\' 'e8m8_EU_noise_' cell2mat(DataName) ],'Accuracy','optK','NumberOfSelectedF','avgSizeClusters','selectedF','newDATA','time1','clusters','OLDclusters');

end
end
meanAcc = mean(Accuracy)
%meanAccOrg = mean(AccuracyOrg)
%meanNumOfEval = mean(numOfEvals)
save(['results_SFFS_fold\'  cell2mat(DataName) ] ,'meanAcc','Accuracy','NumberOfSelectedF','selectedF','time1','CVO');

end

