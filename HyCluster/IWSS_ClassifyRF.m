function [ ACC,acc] = IWSS_ClassifyRF( DATA ,CVO)

acc = [];
time=1;
fold=5;
%data = data(1:(numInst-rem(numInst,fold)),:);
y = DATA(:,end);
%DATA = [DATA(:,1:8) DATA(:,end)];
x = DATA(:,1:end-1);

for t = 1 :time % tenTime
    
%CVO = cvpartition(y,'k',fold); % Stratified cross-validation
for i = 1:CVO.NumTestSets
    
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    ytest = y(teIdx);
    ypred=[];

%     mdl = TreeBagger(60,x(trIdx,indxxx),y(trIdx,:),'Method','classification');
%     ypred = predict(mdl , x(teIdx,indxxx));
%     
    %ytest = classify(x(teIdx,:),x(trIdx,:),y(trIdx,:));
    %err(i) = sum(~strcmp(ytest,species(teIdx)));
    Train = DATA(trIdx,:);
    Test = DATA(teIdx,:);
    save train111.txt Train -ascii
    save test111.txt Test -ascii
    
    ArffTrain = convertToArff('train111.txt');
    ArffTest = convertToArff('test111.txt');
    
    % Train a J48 classifier
    classifier = weka.classifiers.trees.RandomForest();
    classifier.buildClassifier(ArffTrain);
    classifiers{i} = classifier;
    
    % Train a naive bayes classifier
%     classifier = weka.classifiers.bayes.NaiveBayes();
%     classifier.buildClassifier(ArffTrain);
%     classifiers{i} = classifier;
    
    % Train a RIPPER classifier
%     classifier =  weka.classifiers.rules.JRip();
%     classifier.buildClassifier(ArffTrain);
%     classifiers{i} = classifier;
    
    % Train a IB1 classifier
%     classifier =  weka.classifiers.lazy.IB1();
%     classifier.buildClassifier(ArffTrain);
%     classifiers{i} = classifier;
    
    % Classify test instances
    numInst2 = ArffTest.numInstances();
    for k=1:numInst2
        
        temp = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
        ypred(k,1) = str2num(char(ArffTest.classAttribute().value((temp)))); % Predicted labels
        
        
    end
    %temptest(i,:) = estimatedTestLabels';
    
    
    acc(i,t)=(length(ytest) - sum(ytest ~= ypred) ) / length(ytest) ;

end

end

ACC = mean(mean(acc));
end

