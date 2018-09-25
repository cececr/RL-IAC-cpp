function evalBoxes
    addpath (genpath('../edges'))
    addpath('../edges/toolbox')

    % dataset variables
    impath = 'rgbd_scenes_data';
    saliencypath = 'saliency_maps/RGBDscenes_saliency';
    bmspath = 'saliency_maps/BMS';
    filename_gt = 'boundingboxes_RGBDscenes_GT.txt';
    filename_seg = 'boundingboxes_RGBDscenes_seg.txt';
%     impath = 'ENSTA-seq_data';
%     saliencypath = 'saliency_maps/ENSTA-seq_saliency';
%     bmspath = 'saliency_maps/BMS/image_data';
%     filename_gt = 'boundingboxes_ROBOT_GT.txt';
%     filename_seg = 'boundingboxes_ROBOT_seg.txt';
    nboxes = 100;

    % methods
    methods = {'EdgeBoxes'; 'EdgeBoxes+Craye'; 'EdgeBoxes+BMS'; ...
        'EdgeBoxes+Craye+segmentation'; 'Craye+segmentation'};
%     methods = { 'EdgeBoxes+Craye 1';'EdgeBoxes+Craye 2'; 'EdgeBoxes+Craye 3';
%                 'EdgeBoxes+Craye 4';'EdgeBoxes+Craye 5'; 'EdgeBoxes+Craye 6'};
    
    % get ground truth
    bbs_gt =  getBoxesFromFile(filename_gt);
    save('RGBD-scenes_GT.mat', 'bbs_gt')
    % get segmentation boxes
    bbs_seg = getBoxesFromFile(filename_seg);
    % init edgeboxes
    [model, opts]=initEdgeBoxes(nboxes);

    plot_rec = [];
    tolerence_threshold = 0;
    experimentBoxes.methods = methods;
    experimentBoxes.names = bbs_gt.names;
    experimentBoxes.boxes = cell(1);
    for m = 1:length(methods)
        cum_rec = zeros(1,nboxes);
        ct = 0;
        experimentBoxes.boxes{m} = cell(1);
        for i=1:length(bbs_gt.boxes)
            dispwaitmsg(i,length(bbs_gt.boxes));
            bbs = get_bbs(methods{m}, bbs_gt.names{i});
            experimentBoxes.boxes{m}{i} = bbs;
            if isempty(bbs)
                continue
            end
            [~, rec] = getPrecRec(bbs_gt.boxes{i}, bbs);
            rec = [rec rec(end)*ones(1, size(cum_rec,2)-size(rec,2))];
            cum_rec = cum_rec + rec;
            ct = ct+1;

        end
        plot_rec = [plot_rec cum_rec'/ct];
    end
    save('RGBD-scenes_boxes.mat', 'experimentBoxes')
    figure(1), plot(plot_rec)
    legend(methods)
    
    function bbs = get_bbs(method,filename)
        bbs = [];
        bbs_s = [];
        try 
            I = imread([impath '/' filename]);
            salmap = imread([saliencypath '/' filename]);
        catch
            disp(['Could not load file ' filename])
            return 
        end
        if strfind(method, 'EdgeBoxes')
            bbs = edgeBoxes(I, model, opts);
        end
        if strfind(method, 'segmentation')
            bbs_s  = segmentationBoxes(bbs_seg, filename);
        end
        if strfind(method, 'Craye')
            integralSalMap = getIntegralMap(salmap, 2);
        end
        if strfind(method, 'BMS')
            salmap = imread([bmspath '/' filename]);
            integralSalMap = getIntegralMap(salmap, 1);
        end
        if ~isempty(strfind(method, 'Craye')) ...
           || ~isempty(strfind(method, 'BMS'))
            bbs = rankBySaliencyScore(bbs,integralSalMap,tolerence_threshold);
            bbs_s = rankBySaliencyScore(bbs_s,integralSalMap,0.2);
        end
        bbs = [bbs_s;bbs(1:end-size(bbs_s,1),:)];
    end

    function draw_boxes(bbs, color)
        figure(2)
        for j=1:size(bbs,1)
            rectangle('Position',bbs(j,1:4), ...
                  'EdgeColor',color)
        end
    end
    function dispwaitmsg(idx, length)
        if mod(idx/length*1000,10)==0
           disp('...') 
        end
    end
end


function bounding_boxes = getBoxesFromFile(filename)
    fid = fopen(filename,'rt');
    bounding_boxes.boxes = cell(1);
    bounding_boxes.names = cell(1);
    bboxframe = [];
    ct= 0;
    while 1
       tline = fgetl(fid);
       if ~ischar(tline)
          break; 
       end
       if ~isempty(strfind(tline, '.png'))
           ct = ct+1;
           bounding_boxes.names{ct+1} = tline;
           bounding_boxes.boxes{ct} = bboxframe;
           bboxframe = [];
       else
           bboxframe = [bboxframe;str2num(tline)];
       end
    end
    bounding_boxes.names = bounding_boxes.names(2:end);
    bounding_boxes.boxes = bounding_boxes.boxes(2:end);
    fclose(fid);
end

function [model, opts] = initEdgeBoxes(nboxes)
    %% load pre-trained edge detection model and set opts (see edgesDemo.m)
    model=load('models/forest/modelBsds'); model=model.model;
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

    %% set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .65;     % step size of sliding window search
    opts.beta  = .7;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 100;  % max number of boxes to detect
end

function bbs = segmentationBoxes(segboxes, filename)
    bbs = [];
    for j=1:length(segboxes.names)
       if strcmp(segboxes.names{j},filename)
           bbs = segboxes.boxes{j};
           bbs = [bbs ones(size(bbs,1),1)];
           % filter small boxes
           if ~isempty(bbs)
               bbs = bbs(bbs(:,3).*bbs(:,4) > 5000,:);
           end
           break
       end
    end
end

function integralSalMap = getIntegralMap(salmap, scale)
    salmap = imresize(salmap, scale);
    salmap = double(salmap)/255;
    salmap = salmap + 0.01*rand(size(salmap));
    salmap = imgaussfilt(salmap,5);
    integralSalMap = integralImage(salmap);
end

function bbs = rankBySaliencyScore(bbs,integralSalMap,threshold)
    if isempty(bbs)
        return
    end
    sal_scores = [];
    for j=1:size(bbs,1)
        score = get_saliency_score(bbs(j,1:4), integralSalMap);
        score = score * (bbs(j,5));
        sal_scores = [sal_scores; score];
    end
    [sal_scores,sortedIdx] = sort(sal_scores, 'descend');
    sortedIdx = sortedIdx(sal_scores > threshold);
    sal_scores = sal_scores(sal_scores>threshold);
    bbs = [bbs(sortedIdx,1:4) sal_scores];
end
