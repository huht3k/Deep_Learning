function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units    (inputSize by Mr. Hu)
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

%% by Mr. Hu
% inputSize: the number of input units at the first layer, BUT NOT used in
% this routine, becasue netconfig contains such messages
% netconfig:   netconfig.inputsize = size(stack{1}.w, 2)
% netconfig:   netconfig.layersizes = {}....


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

%% forward  propagation
%% FP for the hidden layers
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
expXhat = exp(xhat);     %%% z for the last layer
P = expXhat ./ repmat(sum(expXhat, 1), numClasses, 1);    %%% a for the last layer

%%%%%% Back propagation
%% BP for the out layer
softmaxThetaDelta = -(groundTruth - P);
% % %% by Mr. Hu: max likelihood ==> min cost
% % softmaxThetaDelta = groundTruth - P;

%% BP for the hidden layers
dpn{nd}.delta = softmaxTheta' * softmaxThetaDelta .* dpn{nd}.a .* (1 - dpn{nd}.a);
for d = (numel(stack) -1): -1 : 1
    dpn{d}.delta = (stack{d+1}.w)' * dpn{d+1}.delta .* dpn{d}.a .* (1 - dpn{d}.a);
end

%%%%%% gradient
%% gradient for the out layer
softmaxThetaGrad = softmaxThetaDelta * (dpn{nd}.a)' / M + lambda * softmaxTheta;


%% gradient for the hidden layers
stackgrad{1}.w = dpn{1}.delta * data' / M + lambda * stack{1}.w;
stackgrad{1}.b = sum(dpn{1}.delta, 2) / M;

for d = 2 : numel(stack)
    stackgrad{d}.w = dpn{d}.delta * (dpn{d-1}.a)' / M + lambda * stack{d}.w;
    stackgrad{d}.b = sum(dpn{d}.delta, 2) / M;
end

%% cost
%% w for the hidden layer decay
for d = 1 : numel(stack)
    cost = cost + sum(sum(stack{d}.w .^2));
end
%% w for the out layer decay
cost = cost + sum(sum(softmaxTheta .^2));
%% total
cost = -sum(sum(groundTruth .* log(P))) / M + lambda * cost / 2;
% % %% by Mr. Hu: max likelihood ==> min cost
% % cost = sum(sum(groundTruth .* log(P))) / M + lambda * cost / 2;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
