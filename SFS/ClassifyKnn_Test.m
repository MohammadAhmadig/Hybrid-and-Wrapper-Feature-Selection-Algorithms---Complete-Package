function [ ACC ] = ClassifyKnn_Test( Train,Test )


ytest = Test(:,end);

model = fitcknn(Train(:,1:end-1), Train(:,end) ,'NumNeighbors',1);
ACC=(length(ytest) - sum(ytest ~= predict(model, Test(:,1:end-1))) ) / length(ytest) ;

end
