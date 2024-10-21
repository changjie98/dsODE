fre_smoothed = [];
for i = 1:51
    X = fre_pei_reslif(1000:end,i);
    
    % Smooth X in a 1ms window
    window_size = 10;  % The number of time steps corresponding to 1ms
    smoothed_X = smooth(X, window_size);
    fre_smoothed = [fre_smoothed smoothed_X];
end
peaks_all = [];

% Use the findpeaks function to find the maximum point
for i = 1:51
    [peaks, peak_indices] = findpeaks(fre_smoothed(:,i),'MinPeakDistance',90,'MinPeakHeight',10);
    histdiffpeak = histcounts(diff(peaks),-300:5:300);
    peaks_all = [peaks_all histdiffpeak'];
end

data = peaks_all;

data(data < 0) = 0;
data = data / max(max(data));
data = log2(data + 1);
data = flipud(data);

% Create an image and use imagesc instead of heatmap
figure;
subplot('Position', [0.15, 0.55, 0.7, 0.4])
imagesc(data);
colormap(turbo);
colorbar('off');  % First, turn off the default colorbar

% 调整X轴和Y轴标签
ax = gca;
xticks(10000:10:size(data, 2));  % Set the x-axis scale based on data
xticklabels(arrayfun(@(x) sprintf('%d', x * 0.01 + 0.5 - 0.01), xticks, 'UniformOutput', false));  % Convert x-axis labels
yticks(0:20:size(data, 1));  % Set y-axis scale
yticklabels(arrayfun(@(x) sprintf('%d', 60 - x), yticks, 'UniformOutput', false));  % Convert y-axis labels

set(gca, 'FontSize', 16);

% Create and adjust colorbar
cb = colorbar;
pos = get(cb, 'Position');
pos(1) = 0.87;  % Horizontal position, move to the right to avoid contact with the image
pos(3) = 0.04;  % Adjust the width to make it wider
set(cb, 'Position', pos);
% colorbar('off');  % First, turn off the default colorbar


% Process the second set of data and generate an image
fre_smoothed = [];
for i = 1:51
    X = fre_pei_resdif(1000:end,i);
    
    window_size = 10;  
    smoothed_X = smooth(X, window_size);
    fre_smoothed = [fre_smoothed smoothed_X];
end
peaks_all = [];

for i = 1:51
    [peaks, peak_indices] = findpeaks(fre_smoothed(:,i),'MinPeakDistance',90,'MinPeakHeight',10);
    histdiffpeak = histcounts(diff(peaks),-300:5:300);
    peaks_all = [peaks_all histdiffpeak'];
end

data = peaks_all;

data(data < 0) = 0;
data = data / max(max(data));
data = log2(data + 1);
data = flipud(data);

% Create a second image
subplot('Position', [0.15, 0.13, 0.7, 0.4])
imagesc(data);
colormap(turbo);
colorbar('off');  

ax = gca;
xticks(1:10:size(data, 2));  
xticklabels(arrayfun(@(x) sprintf('%.1coarse-grained Markov modelf', x * 0.01 + 0.5 - 0.01), xticks, 'UniformOutput', false)); 
yticks(0:20:size(data, 1)); 
yticklabels(arrayfun(@(x) sprintf('%d', 60 - x), yticks, 'UniformOutput', false)); 

set(gca, 'FontSize', 16);

cb = colorbar;
pos = get(cb, 'Position');
pos(1) = 0.87; 
pos(3) = 0.04; 
set(cb, 'Position', pos);


dim = [0.46, 0.033, 0.01, 0.037];
str = '$p^{EI}$';
annotation('textbox',dim,'interpreter','latex','String',str,'LineStyle','none','fontsize',20);

dim = [0.07, 0.54, 0.03, 0.03];
str = '$\Delta m$';
annotation('textbox',dim,'interpreter','latex','String',str,'LineStyle','none','fontsize',20);

% colorbar;  
