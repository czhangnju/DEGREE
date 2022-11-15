function pp = Evaluation(Clus, gt)

[~,NMI,~] = compute_nmi(Clus, gt);
ACC = Accuracy(Clus, gt);
[f, ~,~]=compute_f(gt, Clus);
[ARI,~,~]=RandIndex(gt, Clus);
[~,~, PUR] = purity(gt,Clus);
pp = [NMI ACC f ARI PUR];
end