function [cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingFeatureCost - given the weights in weightMatrix,
%                          computes the cost and gradient with respect to
%                          the features, given in featureMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);

    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   features given in featureMatrix.     
    %   You may wish to write the non-topographic version, ignoring
    %   the grouping matrix groupMatrix first, and extend the 
    %   non-topographic version to the topographic version later.
    % -------------------- YOUR CODE HERE --------------------
% %     without topographic
% %     grad = 2 * weightMatrix' * (weightMatrix * featureMatrix - patches) / numExamples ...
% %         + lambda * featureMatrix ./ sqrt(featureMatrix .^2 + epsilon);
    grad = 2 * weightMatrix' * (weightMatrix * featureMatrix - patches) / numExamples ...
        + lambda * groupMatrix' * (groupMatrix * featureMatrix .^2 + epsilon).^(-0.5) .* featureMatrix;
    grad = grad(:);
    
% %     without topographic
% %     cost = sum(sum((weightMatrix * featureMatrix - patches) .^ 2)) / numExamples ...
% %         + lambda * sum(sum(sqrt(featureMatrix .^ 2 + epsilon)));
    
    cost = sum(sum((weightMatrix * featureMatrix - patches) .^ 2)) / numExamples ...
        + lambda * sum(sum((groupMatrix * featureMatrix .^ 2 + epsilon).^0.5));
    
end