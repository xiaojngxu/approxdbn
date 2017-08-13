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
%% full-precision DBN model1

% 1(a) train full-precision RBM
% models=dbnFit(data>0.5,[300 200 100],labels);

% 1(b) load a saved full-precision RBM model
load model_l300l200l100.mat;

%% accuracy with free energy

yhat=dbnPredict(models,testdata>0.5);
%print error
fprintf('Classification accuracy is %f\n', 100-sum(yhat~=testlabels)/length(yhat)*100);
%% limit-precision classification with free energy using full-precision DBN model

%%%%parameters%%%%%
numlayer = length(models);
m = 8; % integer part bit-length
n = 4; % hidden neurons fractional part bit-length
l = 16; % class neurons fractional part bit-length

%%%%%%%%%%%%%%%%%%%
models1 = models;
rounding = 0; % rounding down
for layer=1:numlayer  
    models1{layer}.W = limitbit(models1{layer}.W,rounding,m,n);
    models1{layer}.b = limitbit(models1{layer}.b,rounding,m,n);
end
models1{layer}.Wc = limitbit(models1{layer}.Wc,rounding,m,l);
models1{layer}.cc = limitbit(models1{layer}.cc,rounding,m,l);
yhat=dbnPredict(models1,valdata>0.5);
%print error
fprintf('Classification accuracy is %f\n', 100-sum(yhat~=vallabels)/length(yhat)*100);
%% train limit-precision DBN
for i=1:numlayer
    bitlengths{i}.m=m;
    bitlengths{i}.n=n;
end
bitlengths{numlayer}.l=l;

models2 = dbnFit_VP(data>0.5,labels,models1,bitlengths);
yhat = dbnPredict(models2,valdata>0.5);
%print error
fprintf('Classification accuracy is %f\n', 100-sum(yhat~=vallabels)/length(yhat)*100);
