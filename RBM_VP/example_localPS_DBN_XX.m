% Algorithm & Code Development: Xiaojing Xu
% Principal Architect: Srinjoy Das
% Support and Consultation: Jonas Wei-ting Chan; Chih-Yin Kan; Xinyu Zhang; Javier Girado
% Principal Investigator: Professor Ken Kreutz-Delgado

% This script compares criticality orders to random orders by approximating
% different numbers (0 to all) of hidden neurons and compare the accuracy
% curves 
%% load full MNIST dataset
data = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
testdata = loadMNISTImages('t10k-images-idx3-ubyte');
testlabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
data= data';
testdata = testdata';
valdata = data(50001:end,:);
vallabels = labels(50001:end);
data = data(1:50000,:);
labels = labels(1:50000);
%% full-precision DBN model

% 1(a) train full-precision RBM
% models=dbnFit(data>0.5,[500 500 2000],labels);
% save('model_l500l500l2000.mat');

% 1(b) load a saved full-precision RBM model
load model_l500l500l2000.mat;

%%
% calculate error
yhat=dbnPredict(models,testdata>0.5);
%print error
fprintf('Classification accuracy is %f\n', 100-sum(yhat~=testlabels)/length(yhat)*100);

% parameters
numlayer = length(models);
numh=0; %the number of hidden neurons
for layer=1:numlayer;
    numh = numh + size(models{layer}.W,2);
end
numcrit = 1;
%% calculate criticality orders for DBN
c = struct;
for i=1:numcrit
    [c(i).criticality,c(i).order] = criticality_multilayer(data>0.5,labels,models,i);
end
%% hist
for i=1:numcrit
    
    subplot(numlayer+1,length(c),i);
    hist(cell2mat(c(i).criticality));
    title(['criticality distribution' num2str(i)]);
    for j=1:numlayer
        subplot(numlayer+1,length(c),j*length(c)+i);
        hist(c(i).criticality{j});
    end
end
%% calculate and save approximated DBN accuracies with different approximating orders
clear error;
clear error0;
m=8; %Qm.n
n0=8; %critical neurons Qm.n0
n1=-8; %resilient neurons Qm.n1
step = numh/10;
for numc = 0:step:numh
    for ite = 1:100 % run random order approximation for 100 times
        if ((numc==0||numc==numh)&&ite>1)
            error0(2:100,(numc+step)/step)=error0(1,(numc+step)/step);
            break;
        end
        model12 = models;    
        [~,randorder_all] = sort(rand(size(cell2mat(c(1).order))));
        for layer=1:numlayer            
            %%%%%%%%%%% option (a) all layer together random %%%%%%%%
            randorder=randorder_all(1:size(models{layer}.W,2));
            randorder_all(1:size(models{layer}.W,2))=[];

            pick = randorder<=numc;
            %%%%%%%%%%% option (b) layer-wise random %%%%%%%%%%%%%%%%
%             randorder = sort(rand(size(models{layer}.b)));
%             pick = randorder<numc*length(models{layer}.b)/numh;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            n = n0*ones(1,size(models{layer}.W,2));
            n(pick) = n1;
            
            model12{layer}.W = limitbit(model12{layer}.W,0,m,repmat(n,size(model12{layer}.W,1),1));
            model12{layer}.b = limitbit(model12{layer}.b,0,m,repmat(n,size(model12{layer}.b,1),1));
        end
        yhat = dbnPredict(model12,testdata>0.5);
        fprintf('Classification error with c= %d and numc = %d is %f\n', ...
            0,numc,sum(yhat~=testlabels)/length(yhat));
        error0(ite,(numc+step)/step) = sum(yhat~=testlabels)/length(yhat);
    end
    for i=1:numcrit
        model12 = models;
        for layer=1:numlayer
            %%%%%%%%%%% option (a) all layer together random %%%%%%%%
            pick = c(i).order{layer}<=numc;
            %%%%%%%%%%% option (b) layer-wise random %%%%%%%%%%%%%%%%
%             order = sort(c(i).criticality{layer});
%             pick = randorder<numc*length(models{layer}.b)/numh;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            fprintf('%d approx neurons  ',sum(pick));
            n = n0*ones(1,size(models{layer}.W,2));
            n(pick) = n1;
            
            model12{layer}.W = limitbit(model12{layer}.W,0,m,n);
            model12{layer}.b = limitbit(model12{layer}.b,0,m,n);
        end
        yhat = dbnPredict(model12,testdata>0.5);
        fprintf('Classification error with c= %d and numc = %d is %f\n', ...
            i,numc,sum(yhat~=testlabels)/length(yhat));
        error(i,(numc+step)/step) = sum(yhat~=testlabels)/length(yhat);
    end
    fprintf('\n');
end
%% plot criticality vs random curves
yhat = dbnPredict(models,testdata>0.5);
a0 = 1-sum(yhat~=testlabels)/length(yhat); %baseline accuracy

figure;
hold on;
set(gca,'fontsize',20)
for i=1:1
    plot(0:step:numh,(1-error(i,:))/a0*100,'LineWidth',8);
end
plot(0:step:numh,(1-mean(error0,1))/a0*100,':','LineWidth',8);
legend('criticality','random');

xlabel('Number of Approx Neurons');
ylabel('Accuracy (%)');
ylim([0,130]);
ttl = num2str(size(models{1}.W,2));
for layer=2:numlayer
    ttl = [ttl '-' num2str(size(models{layer}.W,2))];
end
title(ttl);