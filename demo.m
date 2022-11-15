
clear; close all;
addpath ./ClusteringMeasure
addpath ./func
addpath ./data

load('MSRC-v1.mat'); 
param.alpha = .001;
param.beta = 1;
param.gamma = 1;
param.lambda = .1;
param.dim = 50; 

[G] = DEGREE(X, param);

cls_num = numel(unique(gt));
perf = [];
for kk =1:10
%due to random initialization in kmeans, the results may have small difference with reported values   
[Clus] = SpectralClustering(G, cls_num);  
result = Evaluation(Clus, gt);
perf = [perf; result];
fprintf("ITER: %d, NMI, ACC, f, ARI, Purity: %.4f, %.4f, %.4f, %.4f, %.4f \n",kk, result(1), result(2), result(3), result(4),result(5));
end
mean(perf)

