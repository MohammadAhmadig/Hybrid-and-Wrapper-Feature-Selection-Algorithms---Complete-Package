function [train] = convertToArff(str)
% Convert the data to ".arff" format

% Read files using Weka

fName = str;
loader = weka.core.converters.MatlabLoader();
loader.setFile(java.io.File(fName));
train = loader.getDataSet();
train.setClassIndex(train.numAttributes()-1 );

% Convert last attribute (class) from numeric to nominal
filter = weka.filters.unsupervised.attribute.NumericToNominal();
filter.setOptions( weka.core.Utils.splitOptions('-R last') );
filter.setInputFormat(train);
train = filter.useFilter(train, filter);

end