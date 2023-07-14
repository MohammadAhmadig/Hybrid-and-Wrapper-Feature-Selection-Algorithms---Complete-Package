function [ fs ] = mySFS_IWSS( Train , keepIn ,classifier,cvo )

xTrain = Train(:,1:end-1);
yTrain = Train(:,end);
num_features = size(xTrain,2);
fs = zeros(1,num_features);
selected = keepIn;
num_nonInitSelected = num_features-sum(keepIn);

if(strcmp(classifier,'C45'))
    if(sum(selected)>0)
        [maxAcc,maxAcclist]=IWSS_ClassifyC45([xTrain(:,selected) yTrain] ,cvo);
    else
        maxAcc=0;
        maxAcclist=[0;0;0;0;0];
    end
    i=0;
    while (i<num_nonInitSelected)

        tempX = xTrain(:,selected);
        tempNonSel=find(ismember(selected,0));
        acc=zeros(1,length(tempNonSel));
        acclist = {};
        for j =1:length(tempNonSel)
            [acc(j),acclist{j}]=IWSS_ClassifyC45([tempX xTrain(:,tempNonSel(j)) yTrain],cvo);
        end
        % baraye general maghale dovom bayad acclist dorost shavad va
        % maxAcvlist shavad
        [tempMaxAcc,IndmaxAcc]=max(acc);
        if((maxAcc < tempMaxAcc) && (sum(acclist{IndmaxAcc} > maxAcclist)>=2))
            selected(tempNonSel(IndmaxAcc))=1;
            tempNonSel(IndmaxAcc)
            maxAcc=acc(IndmaxAcc)
        else
            break;
        end
        i=i+1;
    end
    
end
if(strcmp(classifier,'NB'))
    if(sum(selected)>0)
        [maxAcc,maxAcclist]=IWSS_ClassifyNB([xTrain(:,selected) yTrain],cvo);
    else
        maxAcc=0;
        maxAcclist=[0;0;0;0;0];
    end
    
    i=0;
    while (i<num_nonInitSelected)

        tempX = xTrain(:,selected);
        tempNonSel=find(ismember(selected,0));
        acc=zeros(1,length(tempNonSel));
        for j =1:length(tempNonSel)
            [acc(j),acclist{j}]=IWSS_ClassifyNB([tempX xTrain(:,tempNonSel(j)) yTrain],cvo);
        end
        [tempMaxAcc,IndmaxAcc]=max(acc);
        if((maxAcc < tempMaxAcc) && (sum(acclist{IndmaxAcc} > maxAcclist)>=2))
            selected(tempNonSel(IndmaxAcc))=1;
            tempNonSel(IndmaxAcc)
            maxAcc=acc(IndmaxAcc)
        else
            break;
        end
        i=i+1;
    end
       
end
if(strcmp(classifier,'Knn'))      
    if(sum(selected)>0)
        [maxAcc,maxAcclist]=IWSS_ClassifyKnn([xTrain(:,selected) yTrain],cvo);
    else
        maxAcc=0;
        maxAcclist=[0;0;0;0;0];
    end
    i=0;
    while (i<num_nonInitSelected)

        tempX = xTrain(:,selected);
        tempNonSel=find(ismember(selected,0));
        acc=zeros(1,length(tempNonSel));
%         maxAcc=0;
%         IndmaxAcc=0;
        for j =1:length(tempNonSel)
            [acc(j),acclist{j}]=IWSS_ClassifyKnn([tempX xTrain(:,tempNonSel(j)) yTrain],cvo);
%             if(ClassifyKnn([tempX xTrain(:,tempNonSel(j)) yTrain]) > maxAcc)
%                 IndmaxAcc=j;
%             end
        end
        [tempMaxAcc,IndmaxAcc]=max(acc);
        if((maxAcc < tempMaxAcc) && (sum(acclist{IndmaxAcc} > maxAcclist)>=2))
            selected(tempNonSel(IndmaxAcc))=1;
            tempNonSel(IndmaxAcc)
            maxAcc=acc(IndmaxAcc)
        else
            break;
        end
        i=i+1;
    end
end
if(strcmp(classifier,'SVM'))
    if(sum(selected)>0)
        [maxAcc,maxAcclist]=IWSS_ClassifySVM([xTrain(:,selected) yTrain] ,cvo);
    else
        maxAcc=0;
        maxAcclist=[0;0;0;0;0];
    end
    i=0;
    while (i<num_nonInitSelected)

        tempX = xTrain(:,selected);
        tempNonSel=find(ismember(selected,0));
        acc=zeros(1,length(tempNonSel));
        acclist = {};
        for j =1:length(tempNonSel)
            [acc(j),acclist{j}]=IWSS_ClassifySVM([tempX xTrain(:,tempNonSel(j)) yTrain],cvo);
        end
        % baraye general maghale dovom bayad acclist dorost shavad va
        % maxAcvlist shavad
        [tempMaxAcc,IndmaxAcc]=max(acc);
        if((maxAcc < tempMaxAcc) && (sum(acclist{IndmaxAcc} > maxAcclist)>=2))
            selected(tempNonSel(IndmaxAcc))=1;
            tempNonSel(IndmaxAcc)
            maxAcc=acc(IndmaxAcc)
        else
            break;
        end
        i=i+1;
    end
    
end
if(strcmp(classifier,'RF'))
    if(sum(selected)>0)
        [maxAcc,maxAcclist]=IWSS_ClassifyRF([xTrain(:,selected) yTrain] ,cvo);
    else
        maxAcc=0;
        maxAcclist=[0;0;0;0;0];
    end
    i=0;
    while (i<num_nonInitSelected)

        tempX = xTrain(:,selected);
        tempNonSel=find(ismember(selected,0));
        acc=zeros(1,length(tempNonSel));
        acclist = {};
        for j =1:length(tempNonSel)
            [acc(j),acclist{j}]=IWSS_ClassifyRF([tempX xTrain(:,tempNonSel(j)) yTrain],cvo);
        end
        % baraye general maghale dovom bayad acclist dorost shavad va
        % maxAcvlist shavad
        [tempMaxAcc,IndmaxAcc]=max(acc);
        if((maxAcc < tempMaxAcc) && (sum(acclist{IndmaxAcc} > maxAcclist)>=2))
            selected(tempNonSel(IndmaxAcc))=1;
            tempNonSel(IndmaxAcc)
            maxAcc=acc(IndmaxAcc)
        else
            break;
        end
        i=i+1;
    end
    
end

fs = selected;
end

