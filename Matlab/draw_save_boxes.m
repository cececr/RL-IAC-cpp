function draw_save_boxes

load('RGBD-scenes_boxes.mat')
load('RGBD-scenes_GT.mat');
impath = 'rgbd_scenes_data';
figure(1)
for i=1:length(bbs_gt.names)
    try
   I = imread([impath '/' bbs_gt.names{i}]); 
    catch
        continue;
    end
   I_GT = draw_boxes(bbs_gt.boxes{i}, 'green',I);
   imwrite(I_GT,['gt/' bbs_gt.names{i}])
   subplot(2,2,1), imshow(I_GT);
   boxes = experimentBoxes.boxes{1}{i};
   if isempty(boxes)
       continue
   end
   I_EB = draw_boxes(experimentBoxes.boxes{1}{i}(1:5,:), 'red',I);
   subplot(2,2,2), imshow(I_EB);
   imwrite(I_EB,['edgeboxes/' bbs_gt.names{i}])
   
   % ours
   boxes = experimentBoxes.boxes{2}{i}(1:5,:);
   boxes = boxes(boxes(:,5)> 0.05,:);
%    boxes = boxes(boxes(:,5)> 0.05,:);
   I_craye = draw_boxes(boxes, 'blue',I);
   
   I_craye = draw_boxes(experimentBoxes.boxes{5}{i}, 'yellow',I_craye);
   subplot(2,2,3), imshow(I_craye);
   title(bbs_gt.names{i})
   imwrite(I_craye,['craye/' bbs_gt.names{i}])
   pause
end

function I = draw_boxes(bbs, color,I)
    for j=1:size(bbs,1)
        I = insertShape(I, 'Rectangle',bbs(j,1:4), 'color',color, 'linewidth',4);
%         rectangle('Position',bbs(j,1:4), ...
%               'EdgeColor',color)
    end
