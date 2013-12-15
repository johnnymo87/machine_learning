function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


appLambda = ones(size(theta));
appLambda(1)=0;
pJ = lambda/(2*m) * (appLambda.*theta)'*(appLambda.*theta);
pH = lambda/m * appLambda.*theta;
HX = sigmoid(X*theta); %m x p * p x 1
J = (1/m) * (-(y'*log(HX)) - (1-y)'*log(1- HX)) + pJ; % (m x 1)' * m x 1
grad = (1/m) * ((HX-y)'*X)' + pH; % 1 x 1 (m x 1 - m x 1)' x m x p 




% =============================================================

end
