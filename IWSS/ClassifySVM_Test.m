function [ ACC ] = ClassifySVM_Test( Train,Test )

save train111.txt Train -ascii
save test111.txt Test -ascii
ytest = Test(:,end);
ypred =[];
ArffTrain = convertToArff('train111.txt');
ArffTest = convertToArff('test111.txt');

% Train a J48 classifier
classifier = weka.classifiers.functions.SMO();
classifier.buildClassifier(ArffTrain);
classifiers = classifier;

numInst2 = ArffTest.numInstances();
for k=1:numInst2
    
    temp = classifiers.classifyInstance(ArffTest.instance(k-1));
    ypred(k,1) = str2num(char(ArffTest.classAttribute().value((temp)))); % Predicted labels
    
    
end
%temptest(i,:) = estimatedTestLabels';


ACC=(length(ytest) - sum(ytest ~= ypred) ) / length(ytest) ;
end
