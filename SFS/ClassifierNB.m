function err = ClassifierNB(xTrain, yTrain, xTest, yTest)

    Train = [xTrain yTrain];
    Test = [xTest yTest];
    ypred=[];
    save train4.txt Train -ascii
    save test4.txt Test -ascii
    
    ArffTrain = convertToArff('train4.txt');
    ArffTest = convertToArff('test4.txt');
    

    %Train a naive bayes classifier
    classifier = weka.classifiers.bayes.NaiveBayes();
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

