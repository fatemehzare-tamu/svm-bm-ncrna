function  [accuracy,labelfinal] = findaccuracy4(X,Y,gcell,alpha,T)

correct=0;
wrong=0;

for i=1: length(Y)

    label=[];
    xx=X(i,:);
    yy=Y(i);
%   alphaclassifier=alphaclassifier./(sum(alphaclassifier));
    for k=1:T
                
         label(k) =predict(gcell{1,k},xx);
                
     end
     labelfinal(i)=sign(alpha*label');
 
     if labelfinal(i)==yy
           correct=correct+1;
     else
           wrong=wrong+1;
%            yy
     end
       
end
accuracy=correct/length(Y);
end





