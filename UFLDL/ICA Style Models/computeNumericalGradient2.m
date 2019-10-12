function numgrad = computeNumericalGradient2(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a matrix of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
[row, col] = size(theta);

theta = theta(:);
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPSILON = 1e-4;
e = eye(size(theta, 1));
for i = 1 : size(theta, 1)
    %i
    pTheta = theta + EPSILON * e(:, i);
    nTheta = theta - EPSILON * e(:, i);
    
    pTheta = reshape(pTheta, row, col);
    nTheta = reshape(nTheta, row, col);
    
    numgrad(i) = (J(pTheta) - J(nTheta)) / EPSILON / 2;
end






%% ---------------------------------------------------------------
end
