% Algorithm & Code Development: Xiaojing Xu
% Principal Architect: Srinjoy Das
% Support and Consultation: Jonas Wei-ting Chan; Chih-Yin Kan; Xinyu Zhang; Javier Girado
% Principal Investigator: Professor Ken Kreutz-Delgado
function [criticality,order]=criticality_multilayer(data,label,models,par)
% calculate the criticalities of hidden neurons

%INPUTS:
% data                       ... training data
% label                      ... training label
% models                     ... DBN model
% par                        ... criticality metric choice

%OUTPUTS
% criticality                ... criticality values
% order                      ... a vector indicating the criticality order


numlayer = length(models);

if (min(label)==0)
    label = label+1;
end
labelv = zeros(length(label),10);
for i = 1:length(label)
    labelv(i,label(i))=1;
end

% keep values of each layer i in top{i}
testdata = data;
for i=1:numlayer
    testdata= rbmVtoH(models{i}, testdata);
    top{i}=testdata;
end
if numlayer==1
    a0= values_XX(models{numlayer}, data);
else
    a0= values_XX(models{numlayer}, top{numlayer-1});
end

switch par
    case 1
        
        % L1 = 1/2*(label-a0).^2;
        dL1da = (a0-labelv);
        criticality{numlayer} = dsoftmax(dL1da,a0) * models{numlayer}.Wc;
        criticality_all = criticality{numlayer};
        for i=numlayer-1:-1:1
            criticality{i} = criticality{i+1}.*dsigmoid(top{i+1})*models{i+1}.W';
            criticality_all = [criticality{i} criticality_all];
        end
end

criticality_all = abs(mean(criticality_all,1));

[~,order_all] = sort(criticality_all);
for i=1:numlayer
    ci = size(criticality{i},2);
    criticality{i} = criticality_all(1:ci);
    criticality_all(1:ci)=[];
    order{i} = order_all(1:ci);
    order_all(1:ci)=[];
end
