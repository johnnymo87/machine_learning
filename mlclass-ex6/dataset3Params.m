function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C     = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
[idx2, idx1] = find(true(numel(C), numel(sigma)));
pairs = [reshape(C(idx1), [], 1), reshape(sigma(idx2), [], 1)];
predicts = [zeros(size(pairs,1),3)];
for i = 1:pairs
	keyboard();
	C = pairs(i)(1) ; sigma = pairs(i)(2);
	keyboard();
	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
	keyboard();
	predictions = svmPredict(model, Xval);
	keyboard();
	err = mean(double(predictions ~= yval));
	predict(i) = [C Sigma err];
	keyboard();
	fprintf(['C: %f\tsignma: %f\terror: %f\n'], C, sigma, err)
endfor




% =========================================================================

end
