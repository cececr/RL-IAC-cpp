function score = get_saliency_score(r, integralSalMap)
    if r(3)+r(1) > size(integralSalMap,2)
        r(3) = size(integralSalMap,2)-r(1);
    end
    if r(4)+r(2) > size(integralSalMap,1)
        r(4) = size(integralSalMap,1)-r(2);
    end
    if r(1) <= 0
        r(1) = 1;
    end
    if r(2) <= 0
        r(2) = 1;
    end

    r2(1) = r(1) + round(r(3)/3);
    r2(2) = r(2) + round(r(4)/3);
    
    r2(3) = r(1) + round(2*r(3)/3);
    r2(4) = r(2) + round(2*r(4)/3);
    r = r + [0 0 r(1) r(2)];
    score = (integralSalMap(r(2),r(1))+ integralSalMap(r(4),r(3)) ...
        - integralSalMap(r(2),r(3)) - integralSalMap(r(4),r(1))) ...
         / ((r(4)-r(2))*(r(3)-r(1))); 
    score2 = (integralSalMap(r2(2),r2(1))+ integralSalMap(r2(4),r2(3)) ...
        - integralSalMap(r2(2),r2(3)) - integralSalMap(r2(4),r2(1))) ...
         / ((r2(4)-r2(2))*(r2(3)-r2(1))); 
     score = 10*score2+score;
     score = score/11;
%%%%%%%%%%%%%%%%%ù
% Alternative version
%%%%%%%%%%%%%%%%
% r = r + [0 0 r(1) r(2)];
%     score = (integralSalMap(r(2),r(1))+ integralSalMap(r(4),r(3)) ...
%             - integralSalMap(r(2),r(3)) - integralSalMap(r(4),r(1)));
%     padding = 3;
%     r2 =  r-padding + [0 0 2*padding 2*padding];
%     r2(r2<=0) = 1;
%     if r2(4) > size(integralSalMap,1)
%         r2(4) = size(integralSalMap,1);
%     end
%     if r2(3) > size(integralSalMap,2)
%         r2(3) = size(integralSalMap,2);
%     end
%     score2 = integralSalMap(r2(2),r2(1))+ integralSalMap(r2(4),r2(3)) ...
%             - integralSalMap(r2(2),r2(3)) - integralSalMap(r2(4),r2(1));
%         
%     score = score / ((r(4)-r(2))*(r(3)-r(1))) ...
%             - (score2-score)/((r2(4)-r2(2))*(r2(3)-r2(1))-(r(4)-r(2))*(r(3)-r(1)));
