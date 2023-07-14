function [ fs ] = myGA( Train , keepIn ,classifier )


vector_size = length(keepIn)-sum(keepIn);
if(vector_size == 1)
	if(strcmp(classifier,'C45')==1)
		a=ClassifyC45([Train(:,1:end-1) yTrain]);
		b=ClassifyC45([Train(:,1:end-2) yTrain]);
		if(a>b)
			fs=logical([ones(1,sum(keepIn)) 1]);
		else
			fs=logical([ones(1,sum(keepIn)) 0]);
		end
	elseif(strcmp(classifier,'NB')==1)
		a=ClassifyNB([Train(:,1:end-1) yTrain]);
		b=ClassifyNB([Train(:,1:end-2) yTrain]);
		if(a>b)
			fs=logical([ones(1,sum(keepIn)) 1]);
		else
			fs=logical([ones(1,sum(keepIn)) 0]);
		end
	elseif(strcmp(classifier,'Knn')==1)
		a=ClassifyKnn([Train(:,1:end-1) yTrain]);
		b=ClassifyKnn([Train(:,1:end-2) yTrain]);
		if(a>b)
			fs=logical([ones(1,sum(keepIn)) 1]);
		else
			fs=logical([ones(1,sum(keepIn)) 0]);
		end
	end   
else
population_size = 50;
if(population_size> (2^vector_size))
	population_size = (2^vector_size);
end
population = zeros(population_size,vector_size);
for i = 1:population_size
    population(i,:) = randi(2,1,vector_size)-1;
end

fitnesses = zeros(1,population_size);
for i = 1:population_size
    fitnesses(i) = fitness_function(Train ,population(i,:),keepIn ,classifier);
end
Index_flag = 0;
for generation = 1:25
    generation
    % termination condition
    if Index_flag ~= 0
        goal=population(Index_flag,:);
        generation
    end
    
    parents = Parent_Selection(population , population_size,vector_size,fitnesses);
    children = Crossover(parents,vector_size);
    % 20% mutation probability
    if (rand()) >= 0.8
        children = Mutation(children,vector_size);
    end
    [population,fitnesses] = Survival_Selection(Train,population, children,fitnesses,keepIn ,classifier);
    
    Index_flag = Termination(fitnesses);
end

if Index_flag ~= 0
    fs = logical([ones(1,sum(keepIn)) goal]);
else
    [~,ind] = max(fitnesses);
    goal=population(ind,:);
    fs = logical([ones(1,sum(keepIn)) goal]);
end
    
end    
    
end