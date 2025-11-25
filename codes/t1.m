clear all
close all
clc
%%
load('cod_rna2.mat')
%%
class=[-1,1];
index=[];
  for i=1:length(class)
      
      index1=find(cod_rna(:,1)==class(i));
      l=ceil(length(index1)*0.1);
%       l=ceil(1000);
        index=[index;index1(1:l)];
      
%           index=[index;index1];
      
%       length(index1)
  end
   cod_rna=cod_rna( index,:);  
 %%
 cod_rna(:,2:9)=normalize(cod_rna(:,2:9),1);
%%
clc
REPEAT=50;
accuracy1=zeros(1,REPEAT);
ACCURACY1=cell(1,REPEAT);

for k=1:REPEAT
N=100;
T=10;
n2=20;
q=1.7;
index=randperm(length(cod_rna));

Dtrain=cod_rna(   index(1:ceil(length(cod_rna)*0.7))      ,:);% 66% 24507 32571
Dtest=cod_rna(    index((ceil(length(cod_rna)*0.7)+1):end)      ,:); % 33% 32572
fprintf('k= %d\n',k);

% ACCURACY1{1,k}=zeros(10,3);



[gcell,alpha,ACCURACY1{1,k}]=mysvmbm(Dtrain,n2,q,N,T,length(Dtrain),Dtest);

[accuracy1(k),labelfinal] = findaccuracy4(Dtest(:,2:9),Dtest(:,1),gcell,alpha,T);
[~,~,~,~,MR1(k)] = misclassification_rates(labelfinal,Dtest(:,1));

% [~, ada_test]=boost2(Dtrain(:,2:9),Dtrain(:,1), Dtest(:,2:9),N,T);
% [~,~,~,~,MR2(k)] = misclassification_rates(ada_test,Dtest(:,1)); 

% [~, ada_test]= adaboost2(Dtrain(:,2:9),Dtrain(:,1), Dtest(:,2:9),N,T);
% [~,~,~,~,MR3(k)] = misclassification_rates(ada_test,Dtest(:,1)) ;

end
%%
subplot(2,1,1)
plot(1:30,accuracy1(1:30),'k--o')
title('k-times (k = 30) SVM-BM for Dtest (Cod-rnd dataset)')

ylabel('Accuracy')
subplot(2,1,2)
plot(1:30,MR1,'b--o')
ylabel('MR')
xlabel('k')

%%
plot(1:30,MR1,'b--o')
hold on
plot(1:30,MR2,'k--o')
plot(1:30,MR3,'r--o')
legend('SVM-BM','Boosting','Adaboost')
title('k-times (k = 30) misclassification rates for Cod-rnd dataset')
xlabel('k')
ylabel('MR for Dtest')
%%
p1 = ranksum(MR1,MR2)
p2 = ranksum(MR1,MR3)
p3 = ranksum(MR2,MR3)
m1=mean(MR1)
s1=var(MR1)^0.5
m2=mean(MR2)
s2=var(MR2)^0.5
m3=mean(MR3)
s3=var(MR3)^0.5


%%
clc

[wrong,correct,labelfinal] = findaccuracy4(Dtest(:,2:9),labelfinal,gcell,alpha,3);
%%
clc
 [accuracy,LABEL]  = findaccuracy3(Dtest,gcell,alpha,3,1:5);
 %%
 [TP,FP,FN,TN,MR] = misclassification_rates(labelfinal,Dtest(:,1))
 
 %%
%  close all
 [ada_train, ada_test]= adaboost2(Dtrain(:,2:9),Dtrain(:,1), Dtest(:,2:9),1000);
%%
 [TP,FP,FN,TN,MR] = misclassification_rates(ada_train,Dtrain(:,1)) ;
 MR
 %%
 [TP,FP,FN,TN,MR] = misclassification_rates(ada_test,Dtest(:,1)) ;
 MR
 %%
%  clc
 [ada_train, ada_test]=boost2(Dtrain(:,2:9),Dtrain(:,1), Dtest(:,2:9),100);
%%
t = templateSVM('KernelFunction','linear');
tEnsemble = templateEnsemble('GentleBoost',10,t);
Mdl = fitcsvm(Dtrain(:,2:9),Dtrain(:,1),'Learners',tEnsemble);
% Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners',tEnsemble,...
%                 'Prior','uniform','NumBins',50,'Options',options);

%%

 

    plot(1:10,[8.6164,8.7099,8.7372,8.7747 ,8.7038, 8.7099,8.7727, 8.7665 , 8.6458, 8.6342],'k--o')
% plot(1:10,[12.5347,13.1938,11.4908,9.4931,9.7626 ,10.3201,9.5818,9.1991,9.4195,10.0260],'k--o')
hold on


% stem(1:10,[12.5347,13.1938,11.4908,9.4931,9.7626 ,10.3201,9.5818,9.1991,9.4195,10.0260],'LineStyle','none','MarkerFaceColor','k','MarkerEdgeColor','k','MarkerSize',10)
ylim([8.6 8.8])
xlabel('t')
ylabel('MR(t)')
title('Misclassification rate for 10 SVM classifiers trained by Markov resampling , Normalized features') 
%%
   A= [0.9620  ,  0.8753  ,  0.8747;...
    0.9660  ,  0.8683  ,  0.8681;...
    0.9790  ,  0.8850   , 0.8851;...
    0.9850  ,  0.9041   , 0.9051;...
    0.9820  ,  0.9014   , 0.9024;...
    0.9680  ,  0.8960   , 0.8968;...
    0.9700  ,  0.9034   , 0.9042;...
    0.9920  ,  0.9068   , 0.9080;...
    0.9720  ,  0.9043   , 0.9058;...
    0.9780  ,  0.8990   , 0.8997];
%%
B=  [0.9750 ,   0.9123  ,  0.9138;...
    0.9890   , 0.9113   , 0.9129;...
    0.9940    ,0.9115   , 0.9126;...
    0.9960  ,  0.9114   , 0.9123;...
    0.9990  ,  0.9119   , 0.9130;...
    0.9980  ,  0.9122   , 0.9129;...
    0.9970 ,   0.9115   , 0.9123;...
    0.9950  ,  0.9113   , 0.9123;...
    0.9980  ,  0.9123   , 0.9135;...
    0.9980  ,  0.9126    ,0.9137];
plot(1:10,B(:,1),'b--o')
hold on
plot(1:10,B(:,2),'k--o')
plot(1:10,B(:,3),'r--o')
legend('Dt accuracy','Dtrain accuracy','Dtest accuracy')
ylim([0.9 1])
xlabel('t')
ylabel('Accuracy')
title('10 SVM classifiers trained by Markov resampling , Normalized features') 

%%
plot(1:10,A(:,1),'b--o')
hold on
plot(1:10,A(:,2),'k--o')
plot(1:10,A(:,3),'r--o')
legend('Dt accuracy','Dtrain accuracy','Dtest accuracy')
ylim([0.865 0.995])
xlabel('t')
ylabel('Accuracy')
title('10 SVM classifiers trained by Markov resampling') 
