function wRR = ridgereg(X,y,lamdaRR)

wRR = (X'*X + (lamdaRR/(1-lamdaRR))*eye(size(X,2))) \ (X'*y); 

end

