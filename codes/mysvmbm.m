function [gcell,alpha,ACCURACY1] = mysvmbm(Dtrain,n2,q,N,T,l,Dtest)

index=randperm(l);
x0=Dtrain(index(1:N),2:9);
y0=Dtrain(index(1:N),1);
[LANDA] =regularizationp(x0,y0);

LANDA;
g0 =fitcsvm(x0,y0,'KernelFunction','linear','BoxConstraint',LANDA);

% t = templateSVM('KernelFunction','linear');
gcell=cell(1,T);
pt=cell(1,T);
Dt=cell(1,T);
for i=1:T
    
   pt{1,i}=zeros(1,N+1); 
   Dt{1,i}=zeros(N,9); 
end
% g0 = fitcecoc(D0,y0,'Learners',t,'ClassNames',[0;1;2;3;4;5;6;7;8;9]);
% accuracy=testpredicter(g0,D0,y0,N);
index=randperm(l);
xi(1,1:8)=Dtrain(index(10),2:9);
yi(1)=Dtrain(index(10),1);
t=1;
flag=zeros(1,N+1);
test=0;
while t<(T+1)
%     count=1;
t
%     Dt{1,i}=zeros(N,9);
    i=1;
    n1=0;
flag=zeros(1,N+1);
    while i<(N+1)
        
        
        index=randperm(l);
        xstar=Dtrain(index(1),2:9);
        ystar=Dtrain(index(1),1);
        % pt i+1
        if t==1
            [L1,~]=myloss(g0,xstar,ystar);
            [L2,~]=myloss(g0,xi(i,:),yi(i));
 
        else
           [L1,~] = myloss(gcell{t-1},xstar,ystar);
           [L2,~] = myloss(gcell{t-1},xi(i,:),yi(i)); 
        end
        
        pt{1,t}(i+1)=min(1,exp(-L1)/exp(-L2)); 
        if n1>n2
           
            pt{1,t}(i+1)=min(1,q* pt{1,t}(i+1));

            Dt{1,t}(i,2:9)= xstar;
            Dt{1,t}(i,1)= ystar;
            flag(i)=1;
            i=i+1;
            xi(i,:)=xstar;
            yi(i)=ystar;            
            
            n1=0;
            
        end

        if (pt{1,t}(i+1)==1)&& ((ystar*yi(i))==1)
           
            
        if t==1
            [~,score1]=myloss(g0,xstar,ystar);
            [~,score2]=myloss(g0,xi(i,:),yi(i));
 
        else
           [~,score1] = myloss(gcell{t-1},xstar,ystar);
           [~,score2] = myloss(gcell{t-1},xi(i,:),yi(i)); 
        end

            pt{1,t}(i+1)=(exp(-ystar*score1)/exp(-yi(i)*score2)); 
        end
      if rand(1)< pt{1,t}(i+1)
          Dt{1,t}(i,2:9)=xstar;
          Dt{1,t}(i,1)= ystar;
          flag(i)=1;
          i=i+1;
          xi(i,:)=xstar;
          yi(i)=ystar;
         
          
          n1=0;      
          
      end
      if flag(i)==0
          
         n1=n1+1;
      end
      
    end
    test=test+1;
    [LANDA] =regularizationp(Dt{1,t}(:,2:9),Dt{1,t}(:,1));
%     LANDA;
    gcell{1,t}=fitcsvm(Dt{1,t}(:,2:9),Dt{1,t}(:,1),'KernelFunction','linear','BoxConstraint',LANDA);%train two class 
    [accuracy_train,~] = testpredicter(gcell{1,t},Dtrain(:,2:9),Dtrain(:,1),l);
    [accuracy_Dt,~] = testpredicter(gcell{1,t},Dt{1,t}(:,2:9),Dt{1,t}(:,1),N);
    [accuracy_Dtest,label] = testpredicter(gcell{1,t},Dtest(:,2:9),Dtest(:,1),length(Dtest));
    ACCURACY1(t,1:3)=[accuracy_Dt,accuracy_train,accuracy_Dtest];
    [~,~,~,~,MR(t)] = misclassification_rates(label,Dtest(:,1));
    
    et(t)=(1-accuracy_train);
    alpha(t)=0.5*log( (1-et(t))  / (et(t)) );
    xi=zeros(1,8);
    yi=zeros(1,1);
    xi(1,:)=xstar;
    yi(1)=ystar;
    
    if (alpha(t)>0)
        t=t+1; 
        test=0;
    end
        
end



end



