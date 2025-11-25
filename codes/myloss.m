function [loss,score] = myloss(g,x,y)

[label,score] =predict(g,x);
if length(score)==2
    score=abs(score(:,2));
else
    score=abs(score);
end
score=label*score;


if (score*y)>1
    
    loss=0;
else
    loss=1-score*y;

end

