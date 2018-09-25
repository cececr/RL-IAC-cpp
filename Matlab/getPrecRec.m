% compute precision-recall curve
function [precision, recall] = getPrecRec(gtboxes, estboxes)
    simthresh = 0.7;
    nbgtboxes = size(gtboxes,1);
    nbestboxes =  size(estboxes,1);
    rect_similarity = zeros(nbgtboxes,nbestboxes);
    for i =1:nbgtboxes
        for j = 1:nbestboxes
            rect_similarity(i,j) =...
                get_rect_similarity(gtboxes(i,1:4),estboxes(j,1:4));
        end
    end
    % get tp and fp
    max_sim = max(rect_similarity,[],1);
    fp_list = max_sim <= simthresh;
    fp_list = integralImage(fp_list);
    fp_list = fp_list(2:end,2:end);
    % get fn
    fn_list = zeros(1,nbestboxes);
    for i =1:nbgtboxes
        ind = find(rect_similarity(i,:) > simthresh);
        if ~isempty(ind)
            fn_list(1:ind(1)-1) = fn_list(1:ind(1)-1) + 1;
        else
            fn_list = fn_list + 1;
        end
    end
    
    precision = (nbestboxes-fp_list)/nbestboxes;
    recall = (nbgtboxes-fn_list)/nbgtboxes;
    if isempty(precision)
        recall = ones(1, nbestboxes);
        precision = zeros(1,nbestboxes);
    end