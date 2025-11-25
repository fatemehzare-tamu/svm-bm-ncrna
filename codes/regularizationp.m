function [LANDA] =regularizationp(x0,y0)
% regularization parameters of SVM selected from ( 0.01, 0.1, 1,100, 1000) by the method of 5-fold cross-validation

regular=[0.01,0.1,1,100,1000];
index=randperm(length(y0));
L=ceil(length(index)/5);
xfold1=x0(index(1:L),:);
xfold2=x0(index((L+1):(2*L)),:);
xfold3=x0(index((2*L+1):(3*L)),:);
xfold4=x0(index((3*L+1):(4*L)),:);
xfold5=x0(index((4*L+1):end),:);

yfold1=y0(index(1:L));
yfold2=y0(index((L+1):(2*L)));
yfold3=y0(index((2*L+1):(3*L)));
yfold4=y0(index((3*L+1):(4*L)));
yfold5=y0(index((4*L+1):end));

% test fold5
g1 =fitcsvm([xfold1;xfold2;xfold3;xfold4],[yfold1;yfold2;yfold3;yfold4],'KernelFunction','linear','BoxConstraint',regular(1));%
[accuracy(1),~] = testpredicter(g1,xfold5,yfold5,length(yfold5));

% test fold4
g2 =fitcsvm([xfold1;xfold2;xfold3;xfold5],[yfold1;yfold2;yfold3;yfold5],'KernelFunction','linear','BoxConstraint', regular(2));%
[accuracy(2),~] = testpredicter(g2,xfold4,yfold4,length(yfold4));

% test fold3
g3 =fitcsvm([xfold1;xfold2;xfold4;xfold5],[yfold1;yfold2;yfold4;yfold5],'KernelFunction','linear','BoxConstraint',regular(3));%
[accuracy(3),~] = testpredicter(g3,xfold3,yfold3,length(yfold3));

% test fold2
g4 =fitcsvm([xfold1;xfold3;xfold4;xfold5],[yfold1;yfold3;yfold4;yfold5],'KernelFunction','linear','BoxConstraint',regular(4));%
[accuracy(4),~] = testpredicter(g4,xfold2,yfold2,length(yfold2));


% test fold1
g5 =fitcsvm([xfold2;xfold3;xfold4;xfold5],[yfold2;yfold3;yfold4;yfold5],'KernelFunction','linear','BoxConstraint',regular(5));%
[accuracy(5),~] = testpredicter(g5,xfold1,yfold1,length(yfold1));

[~,m]=max(accuracy);
LANDA=regular(m);

end

