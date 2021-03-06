function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
C_values     = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
smallest_err = 1000000000;

for c_i = 1:size(C_values, 2)
	c_cur = C_values(c_i);
	for s_i = 1:size(sigma_values, 2)
		s_cur = sigma_values(s_i);
		model= svmTrain(X, y, c_cur, @(x1, x2) gaussianKernel(x1, x2, s_cur));
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		fprintf(['C: %f\tsignma: %f\terror: %f\n'], c_cur, s_cur, err);
		if (err < smallest_err),
			smallest_err = err;
			C = c_cur;
			sigma = s_cur;
		endif
	endfor
endfor


% =========================================================================

end
