function dy = dsigmoid(y)
%dy = dy/dx 
%y=sigmoid(x)
dy = y-y.^2;