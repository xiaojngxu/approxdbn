% Algorithm & Code Development: Xiaojing Xu
% Principal Architect: Srinjoy Das
% Support and Consultation: Jonas Wei-ting Chan; Chih-Yin Kan; Xinyu Zhang; Javier Girado
% Principal Investigator: Professor Ken Kreutz-Delgado
function [model, errors] = rbmFit_VP(X, numhid, y, oldmodel, m, n, l, varargin)
%Fit an RBM to discrete labels in y
%This is not meant to be applied to image data
%based on code by Andrej Karpathy
%based on implementation of Kevin Swersky and Ruslan Salakhutdinov

%INPUTS: 
%X              ... data. should be binary, or in [0,1] interpreted as
%               ... probabilities
%numhid         ... number of hidden units
%y              ... List of discrete labels

%additional inputs (specified as name value pairs or in struct)
%nclasses       ... number of classes
%method         ... CD or SML 
%eta            ... learning rate
%momentum       ... momentum for smoothness amd to prevent overfitting
%               ... NOTE: momentum is not recommended with SML
%maxepoch       ... # of epochs: each is a full pass through train data
%avglast        ... how many epochs before maxepoch to start averaging
%               ... before. Procedure suggested for faster convergence by
%               ... Kevin Swersky in his MSc thesis
%penalty        ... weight decay factor
%weightdecay    ... A boolean flag. When set to true, the weights are
%               ... Decayed linearly from penalty->0.1*penalty in epochs
%batchsize      ... The number of training instances per batch
%verbose        ... For printing progress
%anneal         ... Flag. If set true, the penalty is annealed linearly
%               ... through epochs to 10% of its original value

%OUTPUTS:
%model.W        ... The weights of the connections
%model.b        ... The biases of the hidden layer
%model.c        ... The biases of the visible layer
%model.Wc       ... The weights on labels layer
%model.cc       ... The biases on labels layer

%errors         ... The errors in reconstruction at every epoch

%Process options
args= prepareArgs(varargin);
[   nclasses      ...
    method        ...
    eta           ...
    momentum      ...
    maxepoch      ...
    avglast       ...
    penalty       ...
    batchsize     ...
    verbose       ...
    anneal        ...
    ] = process_options(args    , ...
    'nclasses'      , nunique(y), ...
    'method'        ,  'PCD'     , ...
    'eta'           ,  0.1      , ...
    'momentum'      ,  0.5      , ...
    'maxepoch'      ,  5      , ...
    'avglast'       ,  15        , ...
    'penalty'       , 2e-4      , ...
    'batchsize'     , 100       , ...
    'verbose'       , true     , ...
    'anneal'        , false);
rounding = 2; % rounding scheme. 0 = floor, 1 = ceil, 2 = stochastic
r2=0; % rounding scheme for final model
avgstart = maxepoch - avglast;
oldpenalty= penalty;
[N,d]=size(X);

if (verbose) 
    fprintf('Preprocessing data...\n')
end

%Create targets: 1-of-k encodings for each discrete label
u= unique(y);
targets= zeros(N, nclasses);
for i=1:length(u)
    targets(y==u(i),i)=1;
end

%Create batches
numbatches= ceil(N/batchsize);
groups= repmat(1:numbatches, 1, batchsize);
groups= groups(1:N);
groups = groups(randperm(N));
for i=1:numbatches
    batchdata{i}= X(groups==i,:);
    batchtargets{i}= targets(groups==i,:);
end

%fit RBM
numcases=N;
numdims=d;
numclasses= length(u);

W = oldmodel.W;
c = oldmodel.c;
b = oldmodel.b;
Wc = oldmodel.Wc;
cc = oldmodel.cc;

nhstates = rand(batchsize,numhid);      % initialization for PCD
Winc  = zeros(numdims,numhid);
binc = zeros(1,numhid);
cinc = zeros(1,numdims);
Wcinc = zeros(numclasses,numhid);
ccinc = zeros(1,numclasses);

Wavg = W;
bavg = b;
cavg = c;
Wcavg = Wc;
ccavg = cc;
t = 1;
errors=zeros(1,maxepoch);

for epoch = 1:maxepoch
    
	errsum=0;
    if (anneal)
        penalty= oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
    end
    
    for batch = 1:numbatches
		[numcases numdims]=size(batchdata{batch});
		data = batchdata{batch};
		classes = batchtargets{batch};
        
        %go up
        ph = logistic(data*W + classes*Wc + repmat(b,numcases,1));
		phstates = ph > rand(numcases,numhid);
        if (isequal(method,'SML'))
            if (epoch == 1 && batch == 1)
                nhstates = phstates;
            end
        elseif (isequal(method,'CD'))
            nhstates = phstates;
        elseif (isequal(method,'PCD'))
            nhstates = nhstates;
        end
		
        %go down
		negdata = logistic(nhstates*W' + repmat(c,numcases,1));
		negdatastates = negdata > rand(numcases,numdims);
		negclasses = softmaxPmtk(nhstates*Wc' + repmat(cc,numcases,1));
		negclassesstates = softmax_sample(negclasses);
		
        %go up one more time
		nh = logistic(negdatastates*W + negclassesstates*Wc + ... 
            repmat(b,numcases,1));
		nhstates = nh > rand(numcases,numhid);
		
        %update weights and biases
        
        %%%%%%  (set 1) %%%%%
        dW = (data'*ph - negdata'*nh);
        dc = sum(data) - sum(negdata);
        db = sum(ph) - sum(nh);
        dWc = (classes'*ph - negclasses'*nh);
        dcc = sum(classes) - sum(negclasses);
        
        %%%%%% set 0 (original) %%%%%
        %         dW = (data'*ph - negdatastates'*nh);
        %         dc = sum(data) - sum(negdatastates);
        %         db = sum(ph) - sum(nh);
        %         dWc = (classes'*ph - negclassesstates'*nh);
        %         dcc = sum(classes) - sum(negclassesstates);
        Winc = momentum*Winc + eta*(dW/numcases - penalty*W);
        binc = momentum*binc + eta*(db/numcases);
        cinc = momentum*cinc + eta*(dc/numcases);
        Wcinc = momentum*Wcinc + eta*(dWc/numcases - penalty*Wc);
		ccinc = momentum*ccinc + eta*(dcc/numcases);
		W = W + Winc;
		b = b + binc;
		c = c + cinc;
		Wc = Wc + Wcinc;
		cc = cc + ccinc;
		
        if (epoch > avgstart)
            %apply averaging
			Wavg = Wavg - (1/t)*(Wavg - W);
			cavg = cavg - (1/t)*(cavg - c);
			bavg = bavg - (1/t)*(bavg - b);
			Wcavg = Wcavg - (1/t)*(Wcavg - Wc);
			ccavg = ccavg - (1/t)*(ccavg - cc);
			t = t+1;
		else
			Wavg = W;
			bavg = b;
			cavg = c;
			Wcavg = Wc;
            ccavg = cc;
        end
        W = limitbit(W,rounding,m,n);
        cc = limitbit(cc,rounding,m,l);
        b = limitbit(b,rounding,m,n);
        Wc = limitbit(Wc,rounding,m,l);
        %accumulate reconstruction error
        err= sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;
    end
    
    errors(epoch)= errsum;
    if (verbose) 
        fprintf('Ended epoch %i/%i, Reconsruction error is %f\n', ...
            epoch, maxepoch, errsum);
    end
end

Wavg = limitbit(Wavg,r2,m,n);
ccavg = limitbit(ccavg,r2,m,l);
bavg = limitbit(bavg,r2,m,n);
Wcavg = limitbit(Wcavg,r2,m,l);

model.W= Wavg;
model.b= bavg;
model.c= cavg;
model.Wc= Wcavg;
model.cc= ccavg;
model.labels= u;

