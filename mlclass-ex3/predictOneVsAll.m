function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
%size(all_theta) = 10x401 : the i-th row of all_theta corresponds to the classifier 
%                           for label i (here, images of handwritten digits 1-10)
%                           jth column corresponds to sample (e.g., if row 3 corresponds to
%                           the label "3", there will be 401 values that correspond to the 
%                           intercept + 400 pixels
%                           
%size(X) = 5000 x 400(1)    = 5000 samples (examples) of 400 pixels (400 features per sample)
%size(y) = 5000 x 1         = 5000 known values for the corresponding 400 pixels
%Xrows = examples, Xcolumns = 
%p will be i label rows (10) x 5000 samples  

%hint: [1; 3; 1; 2] is a column vector
S = (all_theta*X')'; %5000 x 10
[i,p] = max(S, [], 2) ;% [row maximum, index of row maximum]
fprintf('size of p is, using function')
size(p)


% =========================================================================


end
