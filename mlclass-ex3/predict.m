function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%Theta1 = 25x401
%Theta2 = 10x26
%X=5000x400

% Add ones to the X data matrix
X = [ones(m, 1) X];                %input layer, a(1), 5000 x 401
Z1 = X*Theta1';                    %(5000 x 25)
a2 = [ones(m, 1) sigmoid(Z1)];              %hidden layer, a(2), 5000 x 26
Z3 = a2*Theta2';                   %output layer,       5000 x 10

[i,p] = max(Z3, [], 2) ;% [row maximum, index of row maximum], p = 5000 x 1


% =========================================================================


end
