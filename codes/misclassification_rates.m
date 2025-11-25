function [TP,FP,FN,TN,MR] = misclassification_rates(label,true)
TP=0;
FP=0;
FN=0;
TN=0;

for i=1:length(true)
    
    
    if label(i)==1
        if true(i)==1
            TP=TP+1;
        else
            FP=FP+1;
        end
    else
        if true(i)==1
            FN=FN+1;
        else
            TN=TN+1;
        end 
    end
end
 MR= (FP+FN)/(FP+FN+TN+TP);           
 MR=MR*100;           
    
end




