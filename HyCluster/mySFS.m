function [ fs ] = mySFS( Train , keepIn ,classifier )

xTrain = Train(:,1:end-1);
yTrain = Train(:,end);
num_features = size(xTrain,2);
fs = zeros(1,num_features);
selected = keepIn;
num_nonInitSelected = num_features-sum(keepIn);

if(strcmp(classifier,'C45'))
    if(sum(selected)>0)
        maxAcc=ClassifyC45([xTrain(:,selected) yTrain]);
    else
        maxAcc=0;
    end
    i=0;
    while (i<num_nonInitSelected)

        tempX = xTrain(:,selected);
        tempNonSel=find(ismember(selected,0));
        acc=zeros(1,length(tempNonSel));
        for j =1:length(tempNonSel)
            acc(j)=ClassifyC45([tempX xTrain(:,tempNonSel(j)) yTrain]);
        end
        
        [tempMaxAcc,IndmaxAcc]=max(acc);
        if(maxAcc < tempMaxAcc)
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
        maxAcc=ClassifyNB([xTrain(:,selected) yTrain]);
    else
        maxAcc=0;
    end
    
    i=0;
    while (i<num_nonInitSelected)

        tempX = xTrain(:,selected);
        tempNonSel=find(ismember(selected,0));
        acc=zeros(1,length(tempNonSel));
        for j =1:length(tempNonSel)
            acc(j)=ClassifyNB([tempX xTrain(:,tempNonSel(j)) yTrain]);
        end
        [tempMaxAcc,IndmaxAcc]=max(acc);
        if(maxAcc < tempMaxAcc)
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
        maxAcc=ClassifyKnn([xTrain(:,selected) yTrain]);
    else
        maxAcc=0;
    end
    i=0;
    while (i<num_nonInitSelected)

        tempX = xTrain(:,selected);
        tempNonSel=find(ismember(selected,0));
        acc=zeros(1,length(tempNonSel));
%         maxAcc=0;
%         IndmaxAcc=0;
        for j =1:length(tempNonSel)
            acc(j)=ClassifyKnn([tempX xTrain(:,tempNonSel(j)) yTrain]);
%             if(ClassifyKnn([tempX xTrain(:,tempNonSel(j)) yTrain]) > maxAcc)
%                 IndmaxAcc=j;
%             end
        end
        [tempMaxAcc,IndmaxAcc]=max(acc);
        if(maxAcc < tempMaxAcc)
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

