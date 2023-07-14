
clear;clc;
%## paths
WEKA_HOME = 'C:\Program Files\Weka-3-8';
javaaddpath('\weka.jar');
K = 10;% kFoldm
Datasets = { 'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat', 'pixraw10P.mat',...
    'GLI-85.mat', 'CLL_SUB_111.mat' , 'SMK_CAN_187.mat','TOX_171.mat'};
	
Classifiers = {'Knn','C45','NB','SVM','RF'};

for DataName = Datasets
    
	HOME = 'dataset_new\';
	load([HOME, cell2mat(DataName)]);
	dataHOME = 'results\';
	load([dataHOME, cell2mat(DataName)],'CVO');

	% normalize data
	X=zscore(X,1);
	
	% define variables
	OrgData =[X Y];
	AccuracyOrg = zeros(5,K);
	Accuracy = zeros(5,K);
	NumberOfSelectedF = zeros(5,K);
	NumberOfSelectedFOrg = zeros(5,K);
	avgSizeClusters = zeros(5,K);
	time1 = zeros(5,K);
	numOfEvals = zeros(5,K);
	% Stratified cross-validation
	%CVO = cvpartition(Y,'k',K); 
	for fold = 1:CVO.NumTestSets
		trIdx = CVO.training(fold);
		teIdx = CVO.test(fold);
		data = OrgData(trIdx,:);
		
		numAttrOriginal = size(data,2);
		maxK =50;
		MaxIter =5;

		%%%%%%%%%%%%%%%% Filter Step %%%%%%%%%%%%%%%%
		NodesSU =[];
		for i = 1 : size(data,2)-1
			NodesSU(i) = (2*mi(data(:,i),data(:,end)))/(h(data(:,i)) + h(data(:,end)));
		end
		[~,sortedSUClass] = sort(NodesSU,'descend');% first is max

		% remove irrelevant features 
		TETA = 0.095;
		relevantSize = floor(sqrt(numAttrOriginal-1) * log(numAttrOriginal-1));

		irrelevantFeatures = sortedSUClass(relevantSize+1:end);
		relevantFeatures = sortedSUClass(1:relevantSize);
		data = [data(:,relevantFeatures) data(:,end)];
		NodesSU = NodesSU(relevantFeatures);
		
		%%%%%%%%%%%%%%%% Feature Clustering %%%%%%%%%%%%%%%%
		%%%%%%%%%%% three methods: 1-KMEANS 2-SNN 3-Spectral Clustering
		
		%%%%%%%%%%% KMEANS %%%%%%%%%%
		[IDX,~,~,optK]=kmeans_opt(data(:,1:end-1)' , maxK);
		IDX;
		%%%%%%%%%%% KMEANS %%%%%%%%%%

		%%%%%%%%%%% SNN %%%%%%%%%%
		% KK=12;
		% minPoints=8;%round(0.7*KK);
		% ep=8;%round(0.5*KK);
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
		% Features_Graph = graph(A);
		% % detect clusters
		% TreeArray = conncomp(Features_Graph);
		% Trees = unique(TreeArray);
		% numOfTree = length(Trees);% Number of Tree(Clusters) - num of connected components
		% clusters ={};
		% sortedInerCluster ={};
		% features =[];clusterSizes =[];
		% for i = 1: numOfTree
		%     [~,idx] = find(ismember(TreeArray,i));
		%     if(~isempty(idx))
		%     clusters{i} = idx;
		%     clusterSizes(i) = length(idx);
		%     temp0 = NodesSU(idx);
		%     [~,max0] = max(temp0);
		%     [~,sortedInerCluster{i}] = sort(temp0,'descend');
		%     features(i) = idx(max0);% nemayandehaye har cluster
		%     end
		% end
		% optK=size(clusters,2);
		%%%%%%%%%%% SNN %%%%%%%%%%

		
		% %%%%%%%%%%% Spectral %%%%%%%%%%
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
		% Adjacency_dist = 1./Adjacency;%%%%
		% 
		% numOfEig = 6;
		% [IDX,~,~,optK]=SpectralClustering(Adjacency ,numOfEig ,maxK ,3);
		%%%%%%%%%%% Spectral %%%%%%%%%%
		
		%%%%%%%%%%%%%%%% Feature Clustering, Hierarchical
		%optK=10;
		% myfunc = @(X)(linkage( X , 'single' , '@func_su'));
		% eva = evalclusters(data',myfunc,'silhouette','KList',[1:maxK]);
		% IDX=eva.OptimalY;
		% optK = eva.OptimalK;
		%[ IDX,optK ] = Clustering( data' , maxK);
		
		%%%%%%%%%%% make graph and plot graph and Minimum spanning tree
		% Features_Graph = graph(round(Adjacency_dist,4));
		% %p = plot(Features_Graph,'EdgeLabel',Features_Graph.Edges.Weight);
		% 
		% [T,pred] = minspantree(Features_Graph,'Method','dense') ; % sparse for kruskal and dense for prim
		% %highlight(p,T)
		% 
		% %Threshold
		% sigma = 0.09;
		% Thresh = mean(1./table2array(T.Edges(:,2)));% - (sigma*std(1./table2array(T.Edges(:,2))));
		% %Thresh = mean(nonzeros(triu(Adjacency))) - (sigma*std(nonzeros(triu(Adjacency))));
		% % or mean and std of adjasency matrix
		% links = table2array(T.Edges(:,1));
		% minEdges = zeros(length(NodesSU),1);
		% for i =1:length(NodesSU)
		%     indx1 = find(links(:,1)==i);
		%     if(length(indx1)~=0)
		%         minEdges(i)=min(1./table2array(T.Edges(indx1,2)));
		%     end
		%     indx1 = find(links(:,2)==i);
		%     if(length(indx1)~=0)
		%         minEdges(i)=min(1./table2array(T.Edges(indx1,2)));
		%     end
		% end

		% detect clusters and inner sort clusters
		clusters ={};
		sortedInerCluster ={};
		features =[];clusterSizes =[];
		for i = 1: optK
			[~,idx] = find(ismember(IDX(1:end)',i));
			if(~isempty(idx))
			clusters{i} = idx;
			clusterSizes(i) = length(idx);
			temp0 = NodesSU(idx);
			[~,max0] = max(temp0);
			[~,sortedInerCluster{i}] = sort(temp0,'descend');
			features(i) = idx(max0);
			end
		end
		OLDclusters=clusters;
		[~,sortedCluster] = sort(NodesSU(features),'descend');
		numOfClusters = size(clusters,2);
		y = data(:,end);

		%%%%%%%%%%%%%%%% Wrapper Step %%%%%%%%%%%%%%%%
		for classifierID = 1:5
			tic;
			numOfEval = 0;
			selectedF =[];results = [];
			for i = 1: numOfClusters
				index = sortedCluster(i);% strat from best cluster to worst cluster
				MaxIter = size(clusters{index},2);
				if(MaxIter > 40)
					MaxIter = floor(sqrt(size(clusters{index},2)));
				end
				newCluster = [selectedF clusters{index}(sortedInerCluster{index})];%each cluster strat from best feature to worst feature
				keepIn = [ones(1,length(selectedF)) zeros(1,length(clusters{index}))];
				keepIn = logical(keepIn);
				x = data(:,newCluster);%%%%%%%%
				c = cvpartition(y,'k',5);
				opts = statset('display','iter','MaxIter',MaxIter);
				if(classifierID == 1)
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

				selectedF = newCluster(find(fs));
				selectedF
				
				% calculate num of evaluation
				numOfselected = sum(keepIn ~= fs);
				if(numOfselected>=1)
					clusterSize = size(clusters{index},2);
					for nEval = 1 : numOfselected
						numOfEval = numOfEval + clusterSize;
						clusterSize = clusterSize - 1;
					end
				end
			end

			if(classifierID == 1)
				Accuracy(classifierID,fold) = ClassifyKnn_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
				NumberOfSelectedF(classifierID,fold) = length(selectedF);
				time1(classifierID,fold) = toc;
			elseif(classifierID == 2)
				Accuracy(classifierID,fold) = ClassifyC45_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
				NumberOfSelectedF(classifierID,fold) = length(selectedF);
				time1(classifierID,fold) = toc;
			elseif(classifierID == 3)
				Accuracy(classifierID,fold) = ClassifyNB_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
				NumberOfSelectedF(classifierID,fold) = length(selectedF);
				time1(classifierID,fold) = toc;
			elseif(classifierID == 4)
				Accuracy(classifierID,fold) = ClassifySVM_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
				NumberOfSelectedF(classifierID,fold) = length(selectedF);
				time1(classifierID,fold) = toc;
			elseif(classifierID == 5)
				Accuracy(classifierID,fold) = ClassifyRF_Test([data(:,selectedF) data(:,end)],[OrgData(teIdx,relevantFeatures(selectedF)) , Y(teIdx)]);
				NumberOfSelectedF(classifierID,fold) = length(selectedF);
				time1(classifierID,fold) = toc;
			end
			
		end
	end
	meanAcc = mean(Accuracy)
	meanNumOfEval = mean(numOfEvals)
	save(['results\'  cell2mat(DataName) ] ,'meanAcc','selectedF','time1','CVO','meanNumOfEval','NumberOfSelectedF');

end

