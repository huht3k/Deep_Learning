function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)

    weightMatrix = reshape(theta, numFeatures, visibleSize);
    
    cost = 0;
    grad = zeros(numFeatures, visibleSize);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------     
    
      numExamples = size(patches, 2);
    
    grad = 2 * weightMatrix * (weightMatrix' * weightMatrix * patches - patches) * patches' ...
        + 2 * weightMatrix * patches * (weightMatrix' * weightMatrix * patches - patches)';
    
    % L1-regularisation epsilon |Wx| ~ sqrt((Wx).^2 + epsilon)
    grad = grad + (weightMatrix * patches ./ sqrt((weightMatrix * patches) .^2 + epsilon)) * patches';
    grad = grad / numExamples;
    
    
    cost = sum(sum((weightMatrix' * weightMatrix * patches - patches) .^ 2));
    
    % L1-regularisation epsilon |Wx| ~ sqrt((Wx).^2 + epsilon)
   cost = cost + sum(sum(sqrt((weightMatrix * patches) .^ 2 + epsilon)));
   cost = cost / numExamples;
    
    
    grad = grad(:);
    

end

