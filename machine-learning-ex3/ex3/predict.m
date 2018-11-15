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

alpha0 = ones(size(X, 1), 1);
alpha1 = [alpha0, X];
Z2 = alpha1 * Theta1';
alpha2 = sigmoid(Z2);
alpha2 = [ones(size(alpha1, 1), 1), alpha2];
Z3 = alpha2 * Theta2';
alpha3 = sigmoid(Z3);
[pval, p] = max(alpha3, [], 2);





% =========================================================================


end
