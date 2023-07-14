clear;
clc
Datasets = {'orlraws10P.mat','pixraw10P.mat','warpPIE10P.mat','warpAR10P.mat','TOX_171.mat', 'SMK_CAN_187.mat','GLI-85.mat','CLL_SUB_111.mat'};%,'BASEHOCK.mat','PCMAC.mat''SMK_CAN_187.mat',
%Datasets = {'orlraws10P.mat','pixraw10P.mat','warpPIE10P.mat','warpAR10P.mat'};%,'BASEHOCK.mat','PCMAC.mat''SMK_CAN_187.mat',

 %{'BASEHOCK.mat','warpPIE10P.mat','warpAR10P.mat','orlraws10P.mat','pixraw10P.mat','PCMAC.mat'};%'GLI-85.mat' ,'GLA-BRA-180.mat' , 'CLL_SUB_111.mat' , 'SMK_CAN_187.mat','TOX_171.mat'};%, 'GLI-85.mat', 'GLA-BRA-180.mat' CLL_SUB_111.mat smk
i=0;
%dispMat = zeros(6,3);
for DataName = Datasets
    i=i+1;
    %'warpAR10P.mat',
HOME = 'results_SFFS_fold\';
load([HOME,cell2mat(DataName)]);%, 'e8m8_EU_noise_'
dispMat(i,1) = 100*mean(Accuracy(5,:));
dispMat(i,2) = mean(NumberOfSelectedF(5,:));

%dispMat(i,2) = 100*meanAccOrg;
%dispMat(i,3) = meanNumOfEval;

% %dispMat(i,1) = Accuracy;
% %dispMat(i,2) = optK;
% dispMat(i,2) = mean(NumberOfSelectedF);
% %dispMat(i,4) = avgSizeClusters;
% dispMat(i,3) = mean(time1);
% % temp=0;
% % for j=1:size(clusters,2)
% %     s=size(clusters{j},2);
% %     temp=temp+((s*(s+1))/2);
% % end
% % dispMat(i,4) = temp;
end
%disp('Accuracy  NumberOfSelectedF   time1')
mean(dispMat(:,1))
mean(dispMat(:,2))
%mean(dispMat(:,3))
%dispMat(:,3)=round(dispMat(:,3),1);
%dispMat(:,1)=round(dispMat(:,1),4);


