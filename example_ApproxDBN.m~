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
% models=dbnFit(data>0.5,[300 200 100],labels);

% 1(b) load a saved full-precision RBM model
load model_l300l200l100.mat;
%% ApproxDBN - Design space exploration: iteratively retraining with criticality
errors = [0.5 5 10];
crits = [0 4];
for ite=1:5
for ierr = 1:length(errors)
    for icrit = 1:length(crits)
        %baseline accuracy
        yhat=dbnPredict(models,valdata>0.5);
        acc = 100-100*sum(yhat~=vallabels)/length(yhat);
        %parameters
        err=errors(ierr); %maximum relative error reduction
        minAcc = acc * (1-err/100);
        crit = crits(icrit); % 1 - using criticality metric 1; 0 - using random selection
        
        [models3,bitlengths3,track3]=findApproxDBN(models,minAcc,data,labels,valdata,vallabels,crit);
        fprintf('Sum of bitlength: %f\n', sumbits(bitlengths3,models3,1));
        save(['ApproxDBN_err' num2str(err) '_c' num2str(crit) '_l500l500l2000_' num2str(ite) '.mat'], 'bitlengths3', 'models3', 'track3');
    end
end
end
%% plot the bar graph of bitlength distribution for: ApproxDBN, ApproxDBN without retraining, ApproxDBN without criticality, ApproxDBN without both

%parameters
numlayer = length(models);
yhat=dbnPredict(models,valdata>0.5);
acc = 100-100*sum(yhat~=vallabels)/length(yhat);
numsample = 5;
errors = [0.5  5 10]; % list of error constraints
bls=[0 4 8 12 16 64]; % list of possible bit-lengths in the network

bl = zeros(length(errors),2,numsample,length(bls)); % bitlength with retraining
bl2 = zeros(length(errors),2,numsample,length(bls)); % bitlength without retraining

for errind = 1:length(errors)
    err = errors(errind);
    for crit = 0:1        
        minAcc = acc * (1-err/100);
        a3=[];
        
        for i=1:numsample
            load(['ApproxDBN_err' num2str(err) '_c' num2str(crit) '_l500l500l2000_' num2str(ite) '.mat']);
            
            if size(track3,1)<=1
                bitlengths2 = bitlengths3;
            else
                bitlengths2 = track3{2,3};
            end
            for j=1:length(bls)
                tmps = 0;
                tmps2=0;
                
                for layer = 1:numlayer
                    tmps = tmps + sum(bitlengths3{layer}.m+bitlengths3{layer}.n==bls(j));
                    tmps2 = tmps2 + sum(bitlengths2{layer}.m+bitlengths2{layer}.n==bls(j));
                end
                bl(errind,crit+1,i,j)=tmps;
                bl2(errind,crit+1,i,j)=tmps2;
            end
            
        end
        
    end
end

bl_retrain=reshape(mean(bl,3),[size(bl,1),size(bl,2),size(bl,4)]);
bl_noretrain = reshape(mean(bl2,3),[size(bl2,1),size(bl2,2),size(bl2,4)]);
bl_bar = cat(2,bl_retrain(:,2,:), bl_retrain(:,1,:),bl_noretrain(:,2,:),bl_noretrain(:,1,:));
bl_sum = sum(bl_bar.*repmat(reshape(bls,1,1,length(bls)),[size(bl_bar,1),size(bl_bar,2),1]),3);
bitdist={'0 bit','4 bit','8 bit','12 bit', '16 bit'};
stackLabels={'ApproxDBN','Without criticality','Without retraining','Without criticality and retraining'};
acc = {'0.5','5','10'};
plotBarStackGroups(bl_bar,acc,stackLabels);
set(gca,'FontSize',18);
legend(bitdist,'Orientation','horizental');
ylabel('Accuracy Loss (%)');
xlabel('Number of Neurons');

%% plot the accuracy and the sum of bit-lengths in the design process
load 'AxDBN_newfirstcrit_ite5_err10_c1_l300l200l100_2.mat';
numlayer = length(models);
yhat=dbnPredict(models,valdata>0.5);

% full bit-length
acc = 100-100*sum(yhat~=vallabels)/length(yhat);
sum_bits = 64*600;
% after phase 1
acc = [acc; track3{1,1}];
bitlengths=track3{1,3};
for layer=1:numlayer
    if (length(bitlengths{layer}.n)==1)
        bitlengths{layer}.n=bitlengths{layer}.n*ones(1,size(models{layer}.W,2));
    end
end
sum_bits = [sum_bits; sumbits(bitlengths,models,0)];

% iterations
for ite = 2:size(track3,1)
    tmpmodel=VPmodelDBN(track3{ite-1,2},track3{ite,3});
    yhat=dbnPredict(tmpmodel,valdata>0.5);
    acc = [acc; 100-100*sum(yhat~=vallabels)/length(yhat)];
    sum_bits = [sum_bits; sumbits(track3{ite,3},models,0)];
    %after retraining
    acc = [acc; track3{ite,1}];
    sum_bits = [sum_bits; sum_bits(end)];
end
acc=acc/acc(1)*100;


yyaxis left     
plot(acc,'LineWidth',6);
ylabel('Accuracy (%)');
yyaxis right;
semilogy(sum_bits,':','LineWidth',6);
ylabel('Sum of bit-length');
legend('Accuracy','Sum of bit-length');
set(gca,'FontSize',18);
% yyaxis left   
% plot(90*ones(size(sum_bits)),'--','LineWidth',3);
set(gca,'XTick',1:length(sum_bits));
set(gca,'XTickLabel',{'start','ite1','','ite2','','ite3','','ite4','','ite5','','ite6','','end'});
