function [accuracy,label] = testpredicter(g,X,Y,l)
correct=0;
label=zeros(1,l);
for i=1:l
    x=X(i,:);
    label(i)=predict(g,x);
%     L = loss(g,x,y(i)) 
    if label(i)==Y(i)
        correct=correct+1;
    end

    
end
accuracy=correct/l;

end

