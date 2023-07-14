function [IDX,optK]=SNN2(d,k,ep,minPoints)
% data input. row: object/sample; column: attribute.
% k: nearest neighbor

numSpl=size(d,1);
%%
[KNNidx,Kdist]=knnsearch(d, d, 'K', k+1, 'Distance', 'euclidean');% @func_su

strength=zeros(numSpl,numSpl);
for i=1:numSpl
for j=KNNidx(i,:)
    if(i~=j)
	nni=KNNidx(i,:);
	nnj=KNNidx(j,:);
	shared=intersect(nni,nnj);
	strength(i,j)=length(shared);
	strength(j,i)=strength(i,j);
    end
end
end
strength(strength<2) =0;%%%%%%%%%%%%%%%%
hplot=plot(graph(strength));
%ep = (mean(mean((1./Kdist(:,2:end))))) - mean(std((1./Kdist(:,2:end)))) ;

SNNdensity = sum((strength)>=ep , 2);
corePoints = SNNdensity > minPoints;% har che minp bozorgtar bashe tedade core ha kamtar va ehtemalan tedade clusterha kamtar
nonCores = ismember(corePoints,0);
highlight(hplot,find(corePoints),'NodeColor','g');

%noise points 
noisePoints=[];
for i =1 : length(nonCores)
    if(nonCores(i) == 1 )
        sims = strength(i,corePoints);
        if(max(sims) < ep)% max sim < ep
            nonCores(i) = 0;% noncores mishavad nonCoresNonNoise
            noisePoints(i) =1;
        end
    end
end
highlight(hplot,find(noisePoints),'NodeColor','r');
highlight(hplot,find(nonCores),'NodeColor','y');

%clustering
IDX = zeros(numSpl,1);
IDX(corePoints) = 1:sum(corePoints);
cores = find(corePoints);
for i =1 : length(corePoints)
	indices=[];
    if(corePoints(i) == 1 )
%         temp=strength;
%         temp(i,i)=1000;
        sims = strength(i,corePoints);
        indices = (sims >= ep);
        if(~isempty(indices))
            IDX(ismember(IDX,IDX(cores(indices))))=IDX(i);
        end
        %IDX(cores(indices))=IDX(i);
    end
    
end

for i =1 : length(corePoints)
	indices=[];
    if(corePoints(i) == 1 )
        sims = strength(i,:);
        indices = (sims >= ep);
        if(~isempty(indices))
            IDX(indices)=IDX(i);
        end
    end
end
valid = sum(noisePoints) == sum(ismember(IDX,0))
if(sum(ismember(IDX,0)) > 0)% if noise data exists
    optK = length(unique(IDX)) -1;
else
    optK = length(unique(IDX));
end

end
