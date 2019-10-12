close all;
clear;
clc;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
n = 10;
alpha = 1;
beta = 2;

x0 = [0 : n];
y0 = linspace(0, 1, 100);

[x, y] = meshgrid(x0, y0);

z = factorial(n) * y .^ (x + alpha - 1) .* (1 - y) .^ (n - x + beta -1) ./ factorial(x)  ./ factorial(n -x);

figure;
mesh(x, y, z);
x1 = xlabel('X');
y1 = ylabel('Y');
z1 = zlabel('Z');
set(x1, 'Rotation', 45);
set(y1, 'Rotation', -45);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% # of samples
N = 1000;
x = zeros(1, N);
y = zeros(1, N);

% starting sampling
start = 500;

y1 = 0.5;


for ii = 1 : N
    x1 = binornd(n, y1);          %% binomial distribution
    y1 = betarnd(x1 + alpha, n - x1 + beta);   %% beta distribution
    x(ii) = x1;
    y(ii) = y1;
end

figure;
% plot(x(start : end), y(start : end));
plot(x(start : end), y(start : end), '.');

