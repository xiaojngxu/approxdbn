% Algorithm & Code Development: Xiaojing Xu
% Principal Architect: Srinjoy Das
% Support and Consultation: Jonas Wei-ting Chan; Chih-Yin Kan; Xinyu Zhang; Javier Girado
% Principal Investigator: Professor Ken Kreutz-Delgado
function sumbit = sumbits(bitlengths,models, includeC)
tmpsumbits = 0;

for layer=1:length(bitlengths)
    if length(bitlengths{layer}.n)==1
        bitlengths{layer}.n=bitlengths{layer}.n*ones(1,length(models{layer}.c));
    end
    tmpsumbits = tmpsumbits + sum(bitlengths{layer}.m+bitlengths{layer}.n);
end

if includeC
    if length(bitlengths{layer}.l)==1
        bitlengths{layer}.l=bitlengths{layer}.l*ones(1,length(models{layer}.cc));
    end
    tmpsumbits = tmpsumbits + sum(bitlengths{layer}.m+bitlengths{layer}.l);
end
sumbit = tmpsumbits;