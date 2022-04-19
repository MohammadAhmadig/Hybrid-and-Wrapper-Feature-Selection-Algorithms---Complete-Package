
function err = ClassifierKnn(xTrain, yTrain, xTest, yTest)
        model = fitcknn(xTrain, yTrain ,'NumNeighbors',1); 
        err = sum(predict(model, xTest) ~= yTest); 

end
% 
% CVO = cvpartition(y,'k',4); % Stratified cross-validation
% for i = 1:CVO.NumTestSets
%     trIdx = CVO.training(i);
%     teIdx = CVO.test(i);
%     ytest = y(teIdx);
%     
% %     mdl = TreeBagger(60,x(trIdx,indxxx),y(trIdx,:),'Method','classification');
% %     ypred = predict(mdl , x(teIdx,indxxx));
% %     
%     %ytest = classify(x(teIdx,:),x(trIdx,:),y(trIdx,:));
%     %err(i) = sum(~strcmp(ytest,species(teIdx)));
%     Train = data(trIdx,:);
%     Test = data(teIdx,:);
%     save train.txt Train -ascii
%     save test.txt Test -ascii
%     
%     ArffTrain = convertToArff('train.txt');
%     ArffTest = convertToArff('test.txt');
%     
%     % Train a J48 classifier
%     classifier = weka.classifiers.trees.J48();
%     classifier.buildClassifier(ArffTrain);
%     classifiers{i} = classifier;
%     
%     % Classify test instances
%     numInst = ArffTest.numInstances();
%     for k=1:numInst
%         
%         temp = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
%         ypred(k,1) = str2num(char(ArffTest.classAttribute().value((temp)))); % Predicted labels
%     end
%     %temptest(i,:) = estimatedTestLabels';
%     
%     
%     % x=features, y=binary response. 
%     c = cvpartition(y,'k',10); 
%     opts = statset('display','iter'); 
%     [fs,history] = sequentialfs(fun,x,y,'cv',c,'options',opts);
% 
%     
%     acc(i)=(length(ytest) - sum(ytest ~= ypred) ) / length(ytest) 