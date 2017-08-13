function [class] = values_XX(model, testdata)
% Use RBM Free Energy to calculate the probability of each class given
% input
% class = 10-d vector. class(i) belongs to [0,1], sum_i class(i) = 1
% size(class) = (size(testdata,1), numclasses)
% Last modified: 08-21-2016 by Xiaojing Xu

numclasses = size(model.Wc, 1);
numcases = size(testdata, 1);
F = zeros(numcases, numclasses);

% set every class bit in turn and find -free energy of the configuration
for i=1:numclasses
    X= zeros(numcases, numclasses);
    X(:, i)=1;
    F(:,i) = repmat(model.cc(i),numcases,1).*X(:,i)+ ...
        sum(log(exp(testdata*model.W+ ...
        X*model.Wc+repmat(model.b,numcases,1))+1),2);
end

class = exp(F-mean(mean(F)));
class = class./ repmat(sum(class,2),1, numclasses);

end