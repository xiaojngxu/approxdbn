% Algorithm & Code Development: Xiaojing Xu
% Principal Architect: Srinjoy Das
% Support and Consultation: Jonas Wei-ting Chan; Chih-Yin Kan; Xinyu Zhang; Javier Girado
% Principal Investigator: Professor Ken Kreutz-Delgado
function newmodel = VPmodelDBN (model,bitlength)
% return limit precision model based on bitlength
newmodel = model;     
for i=1:length(model)
    newmodel{i}.W=limitbit(model{i}.W,0,bitlength{i}.m,bitlength{i}.n);
    newmodel{i}.b=limitbit(model{i}.b,0,bitlength{i}.m,bitlength{i}.n);
end
newmodel{i}.Wc=limitbit(model{i}.Wc,0,bitlength{i}.m,bitlength{i}.l);
newmodel{i}.cc=limitbit(model{i}.cc,0,bitlength{i}.m,bitlength{i}.l);