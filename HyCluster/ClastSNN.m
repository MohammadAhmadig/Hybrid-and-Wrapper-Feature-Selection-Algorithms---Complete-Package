clear;clc;
%## paths

Datasets = {'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat'};%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
for DataName = Datasets
    %'warpAR10P.mat', 'pixraw10P.mat', 'GLI-85.mat' ,'GLA-BRA-180.mat' , 'CLL_SUB_111.mat' , 
HOME = 'dataset_new\';
load([HOME , cell2mat(DataName)]);
data =[X Y];
numAttrOriginal = size(data,2);

[IDX,optK]=SNN1(data(:,1:100)',4,2);


end
