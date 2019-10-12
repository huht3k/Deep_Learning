clear;
close all;
clc;

c = 1.0;
n = 4;
alpha = 50;

% theta = [0.01 : 0.01 : 10];
theta = linspace(0.1, 40, 400);

iChi = c * power(theta, -n / 2) .* exp(-alpha ./ theta / 2);


plot(iChi);