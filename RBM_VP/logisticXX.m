% Algorithm & Code Development: Xiaojing Xu
% Principal Architect: Srinjoy Das
% Support and Consultation: Jonas Wei-ting Chan; Chih-Yin Kan; Xinyu Zhang; Javier Girado
% Principal Investigator: Professor Ken Kreutz-Delgado
function[y] = logisticXX(x,varargin)
% Sigmoid function digital implementation
% code by Xiaojing Xu
% based on paper of Alin TISAN, Stefan ONIGA
% last edit: 2-22-2016
% INPUTS:
%  x         ... input of the logistic function. Can be vector or matrix
%  varargin  ... approximation methods. '' for original sigmoid function.
%  Other methods are A-law, AS and PLAN
% 
if(isfi(x))
    x=x.data;
end
if (nargin == 1 || nargin ==2 && strcmp(varargin{1},''))
    y = 1./(1 + exp(-x));
elseif(nargin == 2)
    xn = -abs(x);
    if (strcmp(varargin{1},'simplest'))
        y = (x>-2 & x<2) .* (1+x/2) / 2 + (x>=2);
    elseif(strcmp(varargin{1},'A-law'))
        y =  ( (xn>-8 & xn<=-4) .* (xn + 8) * (1/16/4) + ...
            (xn>-4 & xn<=-2) .* ((xn + 4) * (1/16/2) + 0.0625) + ...
            (xn>-2 & xn<=-1) .* ((xn + 2) * (1/8/1) + 0.125) + ...
            (xn>-1) .* ((xn + 1) * (1/2/2) + 0.25) ) ...
            .* sign(x) * (-1) + (sign(x) + 1)/2 ;
    elseif(strcmp(varargin{1},'AS'))
        y = ((1/2 + (xn-fix(xn))/4 ) .* 2.^fix(xn) ) ...
            .* sign(x) * (-1) + (sign(x) + 1)/2 ;
    elseif(strcmp(varargin{1},'PLAN'))
        xp = -xn;
        y = ((xp >= 5) + ...
            (xp < 5 & xp >= 2.375) .* (0.03125 * xp + 0.84375) + ...
            (xp < 2.375 & xp >= 1) .* (0.125 * xp + 0.625) + ...
            (xp < 1 ) .* (0.25 * xp + 0.5)) ...
            .* sign(x) + (-sign(x) + 1)/2;
    else
        fprintf('wrong inputs');
    end
else
    fprintf('wrong number of inputs');
end