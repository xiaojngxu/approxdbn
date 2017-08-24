% Algorithm & Code Development: Xiaojing Xu
% Principal Architect: Srinjoy Das
% Support and Consultation: Jonas Wei-ting Chan; Chih-Yin Kan; Xinyu Zhang; Javier Girado
% Principal Investigator: Professor Ken Kreutz-Delgado
function output = limitbit(input, roundingopt, m, n)

% represent x in Qm.n format (fixed-point, m bits before the decimal point,
% n bits after

% roundingopt = 0 if rounding down, 1 if rounding up, 2 if stochastic, 3 if
% to the nearest

% scale = 2^(-n)
% data_range = 2^(m-1) for signed number, 2^m for unsigned number
% last modified: 03-21-2017, Xiaojing Xu

if (numel(m)==1)
    m=m*ones(size(input));
elseif (size(m,1)==1)
    m=repmat(m,size(input,1),1);
elseif (size(m,2)==1)
    m=repmat(m,1,size(input,2));
end


if (nargin == 4)
    if (numel(n)==1)
        n=n*ones(size(input));
    elseif (size(n,1)==1)
        n=repmat(n,size(input,1),1);
    elseif (size(n,2)==1)
        n=repmat(n,1,size(input,2));
    end
    scale = 2.^(-n);
    data_range = 2.^(m-1);
    input(input>=data_range)   = data_range(input>=data_range)-scale(input>=data_range)/64; 
    input(input<=-data_range)   = -data_range(input<=-data_range)+scale(input<=-data_range)/64;
else
    scale = 2.^(-m);
    input(input>=1) = 1-scale(input>=1)/2;
    input(input<=0) = scale(input<=0)/2;
end

if(roundingopt==1)
    output = scale .* ceil( input ./ scale);
elseif (roundingopt==3)
    output = scale .* fix( input ./ scale); 
elseif (roundingopt==0)
    output = scale .* floor( input ./ scale);
elseif (roundingopt==2)
    p = rand(size(input));
    output = scale .* floor( input./scale) + scale .* ((input./scale-floor( input ./ scale))>p);
end