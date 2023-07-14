
clear;clc;
%## paths
WEKA_HOME = 'C:\Program Files\Weka-3-8';
javaaddpath('\weka.jar');
K = 10;% kFold
Datasets = {'SMK_CAN_187.mat' };%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
%Datasets = {'CLL_SUB_111.mat','GLI-85.mat','TOX_171.mat', 'SMK_CAN_187.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
%Datasets = {'GLA-BRA-180.mat' , 'CLL_SUB_111.mat' , 'SMK_CAN_187.mat'};%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat',
for DataName = Datasets
    %'warpAR10P.mat', 'pixraw10P.mat',
HOME = 'dataset_new\';
load([HOME, cell2mat(DataName)]);
    
%load('dataset_new\orlraws10P.mat');
X=zscore(X,1);%%%%%%%%%%%%%%%%%%%%%%%%% normalize
OrgData =[X Y];
Accuracy = zeros(1,K);
NumberOfSelectedF = zeros(1,K);
time1 = zeros(1,K);
CVO = cvpartition(Y,'k',K); % Stratified cross-validation
for fold = 1:CVO.NumTestSets
    trIdx = CVO.training(fold);
    teIdx = CVO.test(fold);
    data = OrgData(trIdx,:);
    
numAttrOriginal = size(data,2);
tic;

MaxIter =5;

NodesSU =[];
for i = 1 : size(data,2)-1
    NodesSU(i) = (2*mi(data(:,i),data(:,end)))/(h(data(:,i)) + h(data(:,end)));
    %NodesSU(i) = (2*(h(data(:,end)) - condh(data(:,end),data(:,i)) )/(h(data(:,i)) + h(data(:,end))) );
end
[~,sortedSUClass] = sort(NodesSU,'descend');% first is max

% remove irrelevant features 
TETA = 0.095;

relevantSize = 30;%floor(sqrt(numAttrOriginal-1) * log(numAttrOriginal-1));
irrelevantFeatures = sortedSUClass(relevantSize+1:end);
relevantFeatures = sortedSUClass(1:relevantSize);
% data(:,irrelevantFeatures) = [];
% NodesSU(:,irrelevantFeatures) = [];
NodesSU_dist = 1./NodesSU;


%data(size(data,1)-mod(size(data,1),10)+1:end,:)=[];% delete rows of data for cv
%OrgDATA(size(OrgDATA,1)-mod(size(OrgDATA,1),10)+1:end,:)=[];
%x = data(:,features);
y = data(:,end);
x = data(:,relevantFeatures);%%%%%%%%
c = cvpartition(y,'k',5);
opts = statset('display','iter');
fun = @(xTrain,yTrain,xTest,yTest)(ClassifierC45(xTrain, yTrain, xTest, yTest));
[fs,history] = sequentialfs(fun,x,y,'cv',c,'options',opts);%%%%%
% if(sum(fs) > length(selectedF))% if feature added with wrapper
%     results(1,i) = ClassifySVM([data(:,newCluster(find(fs))) y]);
%     results(2,i) = sum(fs);
% end
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
    %MaxIter = floor(sqrt(BufferSize));
    newCluster = [selectedF irrelevantFeatures(sortedCMI(1:BufferSize))];
    x = data(:,newCluster);
    keepIn = [ones(1,length(selectedF)) zeros(1,BufferSize)];
    keepIn = logical(keepIn);
    c = cvpartition(y,'k',5);
    opts = statset('display','iter');
    fun = @(xTrain,yTrain,xTest,yTest)(ClassifierC45(xTrain, yTrain, xTest, yTest));
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
Accuracy(fold) = ClassifyC45_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,selectedF) , Y(teIdx)]);
%[Accuracy,indAcc]= max(results(1,:));
%selectedF = selectedF(1:results(2,indAcc))
NumberOfSelectedF(fold) = length(selectedF)
time1(fold) = toc
newDATA = data(:,selectedF);
newDATA = [newDATA y];

% save(['results2\' cell2mat(DataName)],'Accuracy','optK','NumberOfSelectedF','avgSizeClusters','selectedF','newDATA','Acc','sizeOfNewSelectedF');

end
meanAcc=mean(Accuracy)
save(['results_C45_SFS_ReRanking_fold\' cell2mat(DataName)],'CVO','meanAcc','Accuracy','NumberOfSelectedF','selectedF','newDATA','time1');
end

