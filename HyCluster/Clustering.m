function [ IDX,optK ] = Clustering( DATA , maxK)

%DATA = load('study/glass.data');

%first criterion kmeans
% for k = 1:maxK
%     k
%     [idx,C,sumd,D] = kmeans(DATA,k);
%     s(k) = sum(sumd);
% end
% plot(s)

% second criterion
% rng('default');  % For reproducibility
eva = evalclusters(DATA,'linkage','silhouette','KList',[1:maxK]);
IDX=eva.OptimalY;
optK = eva.OptimalK;
%'DaviesBouldin' | 'gap' | 'silhouette' CalinskiHarabasz  kmeans  linkage

%[IDX,C] = kmeans(DATA(:,2:end-1),2);

end


