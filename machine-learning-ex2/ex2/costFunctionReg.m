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


[cost, grad2] = costFunction(theta, X, y);
h = sigmoid(X * theta);

shiftTheta = theta(2:size(theta));
regTheta = [0; shiftTheta];

p = (lambda/(2*m)) * regTheta' * regTheta;
J = cost + p;

grad = (1/m) * (X' * (h - y) + lambda * regTheta);

% =============================================================

end
