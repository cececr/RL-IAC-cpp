addpath (genpath('../edges'))
addpath('../edges/toolbox')
% Demo for Edge Boxes (please see readme.txt first).

%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .7;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 100;  % max number of boxes to detect

impath = '/home/thales/Datasets/rosbag-IROS/image_data';
saliencypath = '/home/thales/Datasets/rosbag-IROS/saliency';
files = dir([impath '/*_depth.png']);
for i = 1:length(files)
   files(i).name = strrep(files(i).name,'_depth',''); 
end

%% get segmentation boxes
fid = fopen('/home/thales/Datasets/rosbag-IROS/boundingboxes.txt','rt');
bounding_boxes = cell(1);
bboxframe = [];
ct= 0;
while 1
   tline = fgetl(fid);
   if ~ischar(tline)
      break; 
   end
   if ~isempty(strfind(tline, 'frame no'))
       ct = ct+1;
       bounding_boxes{ct} = bboxframe;
       bboxframe = [];
   else
       bboxframe = [bboxframe;str2num(tline)];
   end
end
fclose(fid);

ct = 0;
for i=1:length(files)
    %% detect Edge Box bounding box proposals (see edgeBoxes.m)
    %I = imread('peppers.png');
    try        
    salmap = imread([saliencypath '/' files(i).name]);
    catch
        continue;
    end
    ct = ct+1;
    salmap = imresize(salmap, 2);
    salmap = double(salmap)/255;
    integralSalMap = integralImage(salmap);
    I = imread([impath '/' files(i).name]);
    tic, bbs=edgeBoxes(I,model,opts); toc
    
    % filter boxes
    sal_scores = [];
    for j=1:size(bbs,1)
        sal_scores = [sal_scores; ...
                      get_saliency_score(bbs(j,1:4), integralSalMap)];
    end
    remove_idx = zeros(1, size(bbs,1));
    for j = 1:size(bbs,1)-1
        for k = j+1:size(bbs,1)
            if remove_idx(j) + remove_idx(k) > 0
                continue
            end
            sim = get_rect_similarity(bbs(j,1:4),bbs(k,1:4));
            if sim > 0.7
                if sal_scores(k)>sal_scores(j)
                    remove_idx(j) = 1;
                else
                    remove_idx(k) = 1;
                end
            end
        end
    end
    %% show evaluation results (using pre-defined or interactive boxes)
    figure(1); subplot(2,1,1),imshow(I);
    for j=1:size(bbs,1)
        if remove_idx(j) > 0
%             rectangle('Position',bbs(j,1:4),'EdgeColor',[1 0 0])
        else
            if sal_scores(j) > 0.5
                rectangle('Position',bbs(j,1:4), ...
                          'EdgeColor',[0 sal_scores(j) 0])
            else
                rectangle('Position',bbs(j,1:4), ...
                          'EdgeColor',[1-sal_scores(j) 0 0])
            end
        end
    end
    
    %% show segmentation bboxes
    bbs = bounding_boxes{ct};
    sal_scores = [];
    for j=1:size(bbs,1)
        sal_scores = [sal_scores; ...
                      get_saliency_score(bbs(j,1:4), integralSalMap)];
    end
    try
    for j=1:size(bbs,1)
        if sal_scores(j) > 0.5
            rectangle('Position',bbs(j,1:4), ...
                      'EdgeColor',[0 0 sal_scores(j)])
        else
            rectangle('Position',bbs(j,1:4), ...
                      'EdgeColor',[1-sal_scores(j) 0 1])
        end
    end
    catch
    end
    %gt=[122 248 92 65; 193 82 71 53; 410 237 101 81; 204 160 114 95; ...
    %  9 185 86 90; 389 93 120 117; 253 103 107 57; 81 140 91 63];

%     if(1), gt='Please select an object box.'; disp(gt); figure(1); imshow(I);
%       title(gt); [~,gt]=imRectRot('rotate',0); gt=gt.getPos(); end
%     gt(:,5)=0; [gtRes,dtRes]=bbGt('evalRes',gt,double(bbs),.7);
%     figure(1); bbGt('showRes',I,gtRes,dtRes(dtRes(:,6)==1,:));
%     title('green=matched gt  red=missed gt  dashed-green=matched detect');
    
    figure(1), subplot(2,1,2), imshow(salmap)
    drawnow

end
