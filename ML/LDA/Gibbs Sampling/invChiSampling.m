 clear;
 close all;
 clc;
 
% samples
N = 5000;
Theta = zeros(1, N);

% p(x->y): U(0, B)
B = 400;
a = 50;%4;    % PI(theta) = c * power(theta, -n / 2) * exp(-a / theta / 2)
n = 4;%5;
thetaX = 1;
Theta(1) = thetaX;
for ii = 2 : N
    while true
        thetaY = rand() * B;  % potential theta through p(x->y)
        alpha = min(power(thetaY / thetaX, -n / 2) * exp( -a / thetaY / 2 + a / thetaX / 2), 1);
%         alpha(x, y) = min(1, PI(y) / PI(x))
%         PI(y) / PI(x) = power(y/x, -n/2) * exp(-a/y/2 + a/x/2)

        u = rand();
        if u <= alpha
            thetaX = thetaY;
            Theta(ii) = thetaY;
            break;
        end
    end
end
     
% limTheta = Theta(1000 : end);
figure;
plot(Theta);
title('inv chi samples');

figure;
hist(Theta, B / 10);
title('histgram');

 
 