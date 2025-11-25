
function [ada_train, ada_test]= adaboost2(Xtrain,Ytrain, Xtest,N1,T)
% AdaBoost function 
% (X_train-> input: training set)
% (Y_train-> target)
% (Xtest-> input: testing set)
% (ada_train-> label: training set)
% (ada_test-> label: testing set)

N=size(Xtrain,1);

D=(1/N)*ones(N,1);

Classifiers=T;

svm_in=cell(1,Classifiers);
for t=1:Classifiers
%     figure()
%     plot(D)
   index = randsample(1:N,N1,true,D);
   
    X=Xtrain(index,:);
    Y=Ytrain(index);
    
        % svm 
        svm_in{1,t}=fitcsvm(X,Y,'KernelFunction','linear');
        svm_out=predict(svm_in{1,t}, X);
        h=svm_out;
%         Dt=Dt(length(Dt)+1:end,:);
   
%     h_=[h_ h];
    % weighted error
    correct=0;
    for i=1:length(Y)
        if (h(i)==Y(i))
            correct=correct+1; 
        end  
    end
    accuracy=correct/length(Y);
    e(t)=(1-accuracy)+0.01;
    alpha(t)=0.5*log( (1-e(t))  / (e(t)) );
    % Update weights
    D(index)=D(index).*exp((-1).*Y.*alpha(t).*h);
    D=D./sum(D);



%     % weighted error
%     for i=1:length(Y)
%         if (h_(i,T)~=Y(i))
%             eps(T)=eps(T)+D(i,:); 
%         end  
%     end
%     
%     % Hypothesis weight
%     alpha(T)=0.5*log((1-eps(T))/eps(T));
%     
%     % Update weights
%     D=D.*exp((-1).*Y.*alpha(T).*h);
%     D=D./sum(D);
end
% final vote
for t=1:Classifiers
H(:,t)=predict(svm_in{1,t}, Xtrain);
end

ada_train(:,1)=sign(H*alpha');


% final vote
for t=1:Classifiers
H1(:,t)=predict(svm_in{1,t}, Xtest);
end
ada_test(:,1)=sign(H1*alpha');
% for test set

end
