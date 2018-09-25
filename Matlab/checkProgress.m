function checkProgress
%figure
%ylim([0 1]);
%grid
%filenameList = {
%[ 'OnlineRF_RND_test.txt'],
%[ 'OnlineRF_IAC_test.txt']
%};
%checkProgress_(filenameList);
%return
%close all
figure
ylim([0 1]);
grid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pathname = 'exploration/IROS-RGBDscenes-experiments/';
pathname = 'exploration/IROS-robot-experiments/';
%while 1

root = {'IAC','IAC-nolearn','RLIAC','RLIAC-error','RLIAC-novelty', ...
        'RLIAC-uncertainty','Rmax','RND','RNDmarch','RND-nolearn', 'RLIAC-forward'};
root = {'RND-nolearn','IAC-nolearn','RNDmarch','RND','RLIAC','RLIAC-forward'};
% root = {'RND-nolearn','IAC-nolearn','RLIAC-forward'};
    
hold on
displfactor = 0.001;% 2= normal displacement time 0.001= no displacement time
usealpha = [0 1 0];
for alpha = usealpha
    for i = 1:length(root)
        filenameList = {
        [pathname [root{i} '1_log.txt']], ...
        [pathname [root{i} '2_log.txt']], ...
        [pathname [root{i} '3_log.txt']], ...
        [pathname [root{i} '4_log.txt']], ...
        [pathname [root{i} '5_log.txt']]
            };
        filenameList = {
        [pathname [root{i} '1_log.txt']], ...
        [pathname [root{i} '2_log.txt']]
            };
        % get color 
        hsvcol = [(i-1)/length(root),1,1];
        hsvcol = reshape(hsvcol,1,1,3);
        color = hsv2rgb(hsvcol);
        color = reshape(color,1,3);
        checkAvgStdProgress(filenameList, color,displfactor,alpha);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pathname = '/home/thales/QT-WORKSPACE/RL_IAC_SaliencyLearning-Release/log_files/';
% filenameList = {
% [pathname 'Lab_RLIAC_log.txt']
%     };
% checkAvgStdProgress(filenameList, [0 0 1],2);
% filenameList = {
% [pathname 'Lab_RND_log.txt']
%     };
% checkAvgStdProgress(filenameList, [1 0 0],2);
% 
% filenameList = {
% [pathname 'RND_1_nolearn.txt'], ...
% [pathname 'RND_2_nolearn.txt'], ...
% [pathname 'RND_3_nolearn.txt'], ...
% [pathname 'RND_4_nolearn.txt'], ...
% [pathname 'RND_5_nolearn.txt']
%     };
% checkAvgStdProgress(filenameList, [0 1 0],0);



hold off
% grid
%end


legend(root)

function checkAvgStdProgress(filenameList, color, displfactor,usealpha)
meanError = [];
timescale = [];
timediffs = [];
for i = 1:length(filenameList)
    [errorData timeData] = getErrorFromFile(filenameList{i});
    if length(errorData) > length(meanError) && length(meanError) > 0
        errorData = errorData(1:length(meanError));
    elseif length(errorData) < length(meanError)
        meanError = meanError(1:length(errorData),:);
    end
    timeData = changeTimeScale(timeData,displfactor);
    meanError = [meanError errorData];
    timescale = [timescale timeData];
    timediffs = [timediffs; timeData(2:end)-timeData(1:end-1)];
    
end
% figure, hist(timediffs)
[avgError, stdError] = getMeanAndStd(meanError, timescale);
plotMeanAndStd(avgError, stdError, timescale, color,usealpha);

function timeData = changeTimeScale(timeData,displfactor)
for i = 1:length(timeData)-1
    diff = timeData(i+1)-timeData(i);
   if diff > 1
       diff = diff*(displfactor-1);
       timeData(i+1:end) = timeData(i+1:end) + diff;
   end
end


function [meanError, timescale] = getErrorFromFile(filename)

fid = fopen(filename,'rt');
disp(filename)
keepnextline = false;
meanError = [];
timescale = [];
currentCluster = -1;
prevCluster = -1;
ct = 0;
while 1
    tline = fgetl(fid);
    if ~ischar(tline)
        break
    end
    if keepnextline == true
        error_values = str2num(tline);
        error_values = error_values(~isnan(error_values));
        meanerr = mean(error_values);
        meanError = [meanError ; meanerr];
        keepnextline = false;
    end
    if ~isempty(strfind(tline, 'Region scores'))
        keepnextline = true;
    end
    if ~isempty(strfind(tline, 'Time'))
        timescale = [timescale; str2num(tline(7:end))];
    end
    if ~isempty(strfind(tline, 'Chosen cluster = '))
        currentCluster = str2num(tline(17:end));
        if currentCluster ~= prevCluster
            if usetime == 1
                repmeanerr = meanerr*ones(30,1);
                meanError = [meanError ; repmeanerr];
            end
            ct = ct+1;
        end
        prevCluster = currentCluster;
    end
end
fclose(fid);

function [avgError, stdError] = getMeanAndStd(meanErrors, timescale)
lintimescale = transpose(1:max(timescale(:)));
linAvgError = [];
for i = 1:size(meanErrors,2)
    linAvgError = [linAvgError ...
                   interp1(timescale(:,i),meanErrors(:,i),lintimescale)];
end

avgError = mean(linAvgError,2);
stdError = std(linAvgError,[],2).^1.5;
avgError = avgError(~isnan(avgError));
stdError = stdError(~isnan(stdError));
avgError = smooth(avgError,300);
stdError = smooth(stdError,300);
% avgError = mean(meanErrors,2);
% stdError = std(meanErrors,[],2);

function plotMeanAndStd(avgError, stdError, timescale, color,usealpha)
% x = timescale';
% avgError = smooth(avgError,100);
downsampling = 1;
avgError = avgError(1:downsampling:end);
stdError = stdError(1:downsampling:end);
N = length(avgError);
x = 1:length(avgError(1:N));
y1 = 1-avgError(1:N)+stdError(1:N);
y2 = 1-avgError(1:N)-stdError(1:N);
X = [x , fliplr(x)];
Y = [y1', fliplr(y2')];
if usealpha
stdcolor = color'+3;
stdcolor = stdcolor/max(stdcolor(:));
% fill(X,Y, color)
h = fill(X,Y, stdcolor');
set(h,'EdgeColor','None')
% set(h,'FaceAlpha',0.2)
end
hold on
plot(x,1-avgError(1:N), 'color', color, 'linewidth',2)
% pause
% hold off
