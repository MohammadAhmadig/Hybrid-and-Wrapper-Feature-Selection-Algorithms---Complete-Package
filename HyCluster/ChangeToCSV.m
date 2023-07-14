clear;clc;
%Datasets = {'GLI-85.mat' ,'GLA-BRA-180.mat' , 'CLL_SUB_111.mat' , 'SMK_CAN_187.mat','TOX_171.mat'};%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
Datasets = {'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat','pixraw10P.mat'...
    ,'GLI-85.mat' ,'GLA-BRA-180.mat' ,'CLL_SUB_111.mat' , 'SMK_CAN_187.mat','TOX_171.mat'};%%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
for DataName = Datasets%'orlraws10P.mat','warpAR10P.mat','warpPIE10P.mat','PCMAC.mat','BASEHOCK.mat','pixraw10P.mat'...,'GLI-85.mat' ,'GLA-BRA-180.mat' ,'CLL_SUB_111.mat' , 'SMK_CAN_187.mat',
DataName% 'GLI-85.mat' ,'GLA-BRA-180.mat' ,'CLL_SUB_111.mat' , 'SMK_CAN_187.mat',
HOME = 'dataset_new\';
load([HOME, cell2mat(DataName)]);

data=[X Y];
data = [1:size(data,2) ; data];
csvwrite([HOME  cell2mat(DataName) '.csv'],data);

end
