function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

%% FP for the hidden layers
M = size(data, 2);
dpn = cell(numel(stack), 1);
dpn{1}.z = stack{1}.w * data + repmat(stack{1}.b, 1, M);
dpn{1}.a = sigmoid(dpn{1}.z);
for d = 2 : numel(stack)
    dpn{d}.z = stack{d}.w * dpn{d-1}.a + repmat(stack{d}.b, 1, M);
    dpn{d}.a = sigmoid(dpn{d}.z);
end

%% FP for the out layer
nd = numel(stack);     %% the last hidden layer
xhat = softmaxTheta * dpn{nd}.a;  %% no b

[mmax, pred] = max(xhat);









% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
