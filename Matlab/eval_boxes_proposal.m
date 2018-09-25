% evaluate boxes proposal
addpath (genpath('../edges'))
addpath('../edges/toolbox')

%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .7;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 100;  % max number of boxes to detect

% load file names
% impath = 'rgbd_scenes_data';
% saliencypath = 'saliency_maps/RGBDscenes_segmentation';
impath = 'ENSTA-seq_data';
saliencypath = 'saliency_maps/ENSTA-seq_saliency';
files = dir([impath '/*.png']);


%% get ground truth boxes
fid = fopen('boundingboxes_ROBOT_GT.txt','rt');
% fid = fopen('boundingboxes_RGBDscenes_GT.txt','rt');
bounding_boxes = cell(1);
bboxframe = [];
ct= 0;
while 1
   tline = fgetl(fid);
   if ~ischar(tline)
      break; 
   end
   if ~isempty(strfind(tline, '.png'))
       ct = ct+1;
       bounding_boxes{ct} = bboxframe;
       bboxframe = [];
   else
       bboxframe = [bboxframe;str2num(tline)];
   end
end
fclose(fid);

%% get segmentation boxes
fid = fopen('boundingboxes_ROBOT_seg.txt','rt');
% fid = fopen('boundingboxes_RGBDscenes_seg.txt','rt');
segboxes.boxes = cell(1);
segboxes.names = cell(1);
bboxframe = [];
ct= 0;
while 1
   tline = fgetl(fid);
   if ~ischar(tline)
      break; 
   end
   if ~isempty(strfind(tline, '.png'))
       ct = ct+1;
       segboxes.names{ct+1} = tline;
       segboxes.boxes{ct} = bboxframe;
       bboxframe = [];
   else
       bboxframe = [bboxframe;str2num(tline)];
   end
end
fclose(fid);


ct = 0;
cum_prec = zeros(1, opts.maxBoxes);
cum_rec = zeros(1, opts.maxBoxes);
cum_ct = zeros(1, opts.maxBoxes);

sal_cum_prec = zeros(1, opts.maxBoxes);
sal_cum_rec = zeros(1, opts.maxBoxes);
sal_cum_ct = zeros(1, opts.maxBoxes);

for i=1:length(files)-1
    
    try
        salmap = imread([saliencypath '/' files(i).name]);
    catch
        disp(['file ' saliencypath '/' files(i).name 'does not exist' ])
        continue
    end
    
    % compute edge boxes
    try
        I = imread([impath '/' files(i).name]);
    catch
        disp(['file ' impath '/' files(i).name 'does not exist' ])
        continue
    end
    % if edgeboxes
    tic, bbs=edgeBoxes(I,model,opts); toc
    
    % if segboxes
    segbbs = [];
    for j=1:length(segboxes.names)
       if strcmp(segboxes.names{j},files(i).name)
           segbbs = segboxes.boxes{j};
           segbbs = [segbbs ones(size(segbbs,1),1)];
           if ~isempty(segbbs)
               segbbs = segbbs(segbbs(:,3).*segbbs(:,4) > 5000,:);
           end
           break
       end
    end
    
    
    
    if isempty(bbs)
        continue
    end
    
    [prec, rec] = getPrecRec(bounding_boxes{i+1}, bbs);
    prec = [prec prec(end)*ones(1, size(cum_prec,2)-size(prec,2))];
    rec = [rec rec(end)*ones(1, size(cum_rec,2)-size(rec,2))];
    cum_prec(1:length(prec)) = cum_prec(1:length(prec)) + prec;
    cum_rec(1:length(rec)) = cum_rec(1:length(rec)) + rec;
    cum_ct(1:length(rec)) = cum_ct(1:length(rec))+1;

    % load saliency map

    if length(salmap) < length(I)
        salmap = imresize(salmap, 2);
    end
    salmap = double(salmap)/255;
    salmap = salmap + 0.01*rand(size(salmap));
    salmap = imgaussfilt(salmap,5);
    integralSalMap = integralImage(salmap);
        
    % filter boxes
    sal_scores = [];
    for j=1:size(bbs,1)
        score = get_saliency_score(bbs(j,1:4), integralSalMap);
        score = score * (bbs(j,5));
        sal_scores = [sal_scores; score];
    end
    seg_sal_scores = [];
    for j=1:size(segbbs,1)
        score = get_saliency_score(segbbs(j,1:4), integralSalMap);
        seg_sal_scores = [seg_sal_scores; score];
    end
    
    [~,sortedIdx] = sort(sal_scores, 'descend');
    saliency_bbs = bbs(sortedIdx,1:4);
    
    [~,sortedIdx] = sort(seg_sal_scores, 'descend');
    sortedIdx = sortedIdx(seg_sal_scores > 0.2);
    segbbs = segbbs(sortedIdx,1:4);
    
    figure(1); imshow(I);
    for j=1:size(bbs,1)
        rectangle('Position',bbs(j,1:4), ...
                  'EdgeColor',[0 0 1])
          rectangle('Position',saliency_bbs(j,1:4), ...
                  'EdgeColor',[1 0 0])
    end
    for j=1:size(bounding_boxes{i+1},1)
        rectangle('Position',bounding_boxes{i+1}(j,1:4), ...
                  'EdgeColor',[0 1 0])
    end
    drawnow
    
    if ~isempty(segbbs)
        segbbs = segbbs(:,1:4);
    end
    saliency_bbs = [segbbs;saliency_bbs(1:end-length(segbbs),:)];
    [prec, rec] = getPrecRec(bounding_boxes{i+1}, saliency_bbs);
    prec = [prec prec(end)*ones(1, size(sal_cum_prec,2)-size(prec,2))];
    rec = [rec rec(end)*ones(1, size(sal_cum_rec,2)-size(rec,2))];
    sal_cum_prec(1:length(prec)) = sal_cum_prec(1:length(prec)) + prec;
    sal_cum_rec(1:length(rec)) = sal_cum_rec(1:length(rec)) + rec;
    sal_cum_ct(1:length(rec)) = sal_cum_ct(1:length(rec))+1;
%     disp(cum_rec(1:10))
%     disp(sal_cum_rec(1:10))
%     pause

end
    figure,
    plot([cum_rec./cum_ct;sal_cum_rec./sal_cum_ct]');
    ylabel('precision-recall')
    xlabel('nb boxes')
    legend({'edgeboxes';'edgeboxes+saliency'})
