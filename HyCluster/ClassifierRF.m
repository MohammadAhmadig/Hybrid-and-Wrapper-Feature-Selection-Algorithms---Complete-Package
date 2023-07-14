function err = ClassifierRF(xTrain, yTrain, xTest, yTest)

    Train = [xTrain yTrain];
    Test = [xTest yTest];
    save train.txt Train -ascii
    save test.txt Test -ascii
    
    ArffTrain = convertToArff('train.txt');
    ArffTest = convertToArff('test.txt');
    
    % Train a J48 classifier
    classifier = weka.classifiers.trees.RandomForest();
    %classifier = weka.classifiers.trees.J48();
    classifier.buildClassifier(ArffTrain);
    classifiers = classifier;
    
    % Classify test instances
    numInst = ArffTest.numInstances();
    for k=1:numInst
        
        temp = classifiers.classifyInstance(ArffTest.instance(k-1));
        ypred(k,1) = str2num(char(ArffTest.classAttribute().value((temp)))); % Predicted labels
    end

    err = sum(ypred ~= yTest);
end

