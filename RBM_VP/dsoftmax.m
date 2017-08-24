% Algorithm & Code Development: Xiaojing Xu
% Principal Architect: Srinjoy Das
% Support and Consultation: Jonas Wei-ting Chan; Chih-Yin Kan; Xinyu Zhang; Javier Girado
% Principal Investigator: Professor Ken Kreutz-Delgado
function dy=dsoftmax(dl,y)
%dy = dy/dx 
%y=softmax(x)
dy = [];

for i=1:size(y,1)
    tmpy = y(i,:);
    tmpdy = (diag(tmpy) - tmpy'*tmpy);
    dy = [dy; dl(i,:)*tmpdy];
end