function [ApproxModel, bitlengths,track] = findApproxDBN(modelsfull,minAcc,data,labels,testdata,testlabels,crit)
%Use RBM to predict discrete label for testdata with limit precision
%code by Xiaojing Xu
%last edit: 03-21-2017, Xiaojing Xu

%INPUTS:
% modelsfull     ... full precision dbn model
% minAcc         ... min accuracy allowed
% data           ... training data
% labels         ... training labels
% testdata       ... test data
% testlabels     ... test labels
% crit           ... which criticality order to use, 0 -> random

%OUTPUTS:
% ApproxModel    ... Approximate DBN model
% bitlengths     ... bitlengths for all hidden layers
% track          ... cell array keeping track of error, bithlength, and
% model


%PARAMETERS
% m              ... the integral bit length
% n              ... the hidden units bit length
% l              ... the class units bit length
% dc             ... loop level 2 # resilient neuron decrease step size
% db             ... loop level 3 bit-length decrease step size
m=8;
l = 64-m;
numlayer = length(modelsfull);
numhid = 0;
for i=1:numlayer
    bitlengths{i}.m=m;
    bitlengths{i}.n=l;
    numhid = numhid + size(modelsfull{i}.W,2);
end
dc = numhid/10;
db = 4;
bitlengths{i}.l=l;
track=cell(0); %track the model, bitlength and error in the iteratively retraining process
cnt = 1; %initial index for track

acc = 100-100*sum(dbnPredict(modelsfull,testdata>0.5)~=testlabels)/length(testlabels);
fprintf('Classification accuracy with full bit length is %f\n', acc);

ret = 0; % how many times has retraining been done



% decide the approximating order of hidden neurons
if crit>0
    [c.criticality,c.order] = criticality_multilayer(data>0.5,labels,modelsfull,crit);
else
    [~,randorder_all] = sort(rand(1,numhid));
    for layer=1:numlayer
        c.order{layer}=randorder_all(1:size(modelsfull{layer}.W,2));
        randorder_all(1:size(modelsfull{layer}.W,2))=[];
    end
end

% Phase 1: uniform bit-length reduction% Phase 2: bit-length reduction for hidden units using criticality
% reduce bit-length until accuracy drops below minA
while (acc>=minAcc)
    l = l-db;
    prebitlengths=bitlengths;
    for i=1:numlayer
        bitlengths{i}.n=l;
    end
    bitlengths{i}.l=l;
    tmpmodel = VPmodelDBN(modelsfull,bitlengths);
    acc = 100-100*sum(dbnPredict(tmpmodel,testdata>0.5)~=testlabels)/length(testlabels);
    fprintf('Classification accuracy with uniform bit length %d is %f\n', l+m, acc);
end
bitlengths = prebitlengths;
models1 = VPmodelDBN(modelsfull,bitlengths);

acc = 100-100*sum(dbnPredict(models1,testdata>0.5)~=testlabels)/length(testlabels);
track{cnt,1} = acc;
track{cnt,2} = models1;
track{cnt,3} = bitlengths;
cnt = cnt+1;

for layer=1:numlayer
    bitlengths{layer}.n=bitlengths{layer}.n*ones(1,size(models1{layer}.W,2));
end

% calculate the total bitlength
tmpsumb = sumbits(bitlengths,models1,0);

% Phase 2: bit-length reduction for hidden units using criticality
% reduce bit-length until accuracy drops below minA
% loop level 1: retraining
while (acc>=minAcc)
    sumb = tmpsumb;
    numc = numhid; % set the number of approximated neurons to numhid (the total number of hidden neurons)
    
    
    
    % loop level 2: change the number of resilient neurons
    while numc>1
        % loop level 3: reduce the bitlength of numc neurons
        while (acc>=minAcc)
            % reduce bitlength until accuracy drops below minA
            prebitlengths = bitlengths;
            bitchange=false;
            for layer=1:numlayer
                pick=c.order{layer}<=numc;
                pick(bitlengths{layer}.n+bitlengths{layer}.m<=0)=0;
                if sum(pick)>0
                    bitchange=true;
                end
                bitlengths{layer}.n(pick)=bitlengths{layer}.n(pick)-db;
            end
            
            tmpmodel = VPmodelDBN(models1,bitlengths);
            acc = 100-100*sum(dbnPredict(tmpmodel,testdata>0.5)~=testlabels)/length(testlabels);

            if ~bitchange
                break;
            end
        end
        
        if (acc<minAcc)
            bitlengths=prebitlengths;
        end
        models1 = VPmodelDBN(models1,bitlengths);
        acc = 100-100*sum(dbnPredict(models1,testdata>0.5)~=testlabels)/length(testlabels);

        numc = numc-dc;
    end
    
    ApproxModel=models1;
    tmpsumb = sumbits(bitlengths,models1,0);
    if (sumb-tmpsumb<1)
        break;
    end
    fprintf('%d bits changed \n', sumb-tmpsumb);
    
    models1 = dbnFit_VP(data>0.5,labels,models1,bitlengths);
    acc = 100-100*sum(dbnPredict(models1,testdata>0.5)~=testlabels)/length(testlabels);
    track{cnt,1} = acc;
    track{cnt,2} = models1;
    track{cnt,3} = bitlengths;
    cnt = cnt+1;
    ret = ret+1;
    fprintf('After %d th retraining: Classification accuracy is %f\n', ret, acc);
    
    if crit>0
        [c.criticality,c.order] = criticality_multilayer(data>0.5,labels,models1,crit);
    else
        [~,randorder_all] = sort(rand(1,numhid));
        for layer=1:numlayer
            c.order{layer}=randorder_all(1:size(models1{layer}.W,2));
            randorder_all(1:size(models1{layer}.W,2))=[];
        end
    end
    
end