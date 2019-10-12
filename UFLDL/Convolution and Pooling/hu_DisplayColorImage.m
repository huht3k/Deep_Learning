close all;
clear;
clc;

load stlTrainSubset.mat % loads numTrainImages, trainImages, trainLabels

m = 5;
n = 5;
for i = 1 : m * n
    subplot(m, n, i);
    imshow(trainImages(:,:,:,i));
end