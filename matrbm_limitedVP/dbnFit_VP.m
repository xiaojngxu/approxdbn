function model= dbnFitXX(X, y, oldmodel, bitlengths, varargin)
%fit a DBN to bianry data in X

%INPUTS: 
%X              ... data. should be binary, or in [0,1] interpreted as
%               ... probabilities
%numhid         ... list of numbers of hidden units
%y              ... List of discrete labels

%OUTPUTS:
%model          ... A cell array containing models from all RBM's

%varargin may contain options for the RBM's of this DBN, in row one by one
%for example:
%dbnFit(X, [500,400], opt1, opt2) uses opt1 for 500 and opt2 for 400
%dbnFit(X, [500,400], opt1) uses opt1 only for 500, and defaults for 400

numopts=length(varargin);
H = length(oldmodel);
numhid = [];
for i=1:H
    numhid = [numhid size(oldmodel{i}.W,2)];
end
model=oldmodel;
if H>=2
    
    %train the first RBM on data
    if(numopts>=1)
        model{1}= rbmBB_VP(X, numhid(1),oldmodel{1}, bitlengths{1}.m, bitlengths{1}.n, varargin{1});
    else
        model{1}= rbmBB_VP(X, numhid(1),oldmodel{1}, bitlengths{1}.m, bitlengths{1}.n);
    end
    
    %train all other RBM's on top of each other
    for i=2:H-1
        if(numopts>=i)
            model{i}=rbmBB_VP(model{i-1}.top>0.5, numhid(i), oldmodel{i}, bitlengths{i}.m, bitlengths{i}.n,varargin{i});
        else
            model{i}=rbmBB_VP(model{i-1}.top>0.5, numhid(i), oldmodel{i}, bitlengths{i}.m, bitlengths{i}.n);
        end
        
    end
    
    %the last RBM has access to labels too
    if(numopts>=H)
        model{H}= rbmFit_VP(model{H-1}.top>0.5, numhid(end), y, oldmodel{H}, bitlengths{H}.m, bitlengths{H}.n, bitlengths{H}.l, varargin{H});
    else
        model{H}= rbmFit_VP(model{H-1}.top>0.5, numhid(end), y, oldmodel{H}, bitlengths{H}.m, bitlengths{H}.n, bitlengths{H}.l);
    end
else
    
    %numhid is only a single layer... but we should work anyway
    if (numopts>=1)
        model{1}= rbmFit_VP(X, numhid(1), y, oldmodel{1}, bitlengths{1}.m, bitlengths{1}.n, bitlengths{1}.l, varargin{1});
    else
        model{1}= rbmFit_VP(X, numhid(1), y, oldmodel{1}, bitlengths{1}.m, bitlengths{1}.n, bitlengths{1}.l);
    end
end    

