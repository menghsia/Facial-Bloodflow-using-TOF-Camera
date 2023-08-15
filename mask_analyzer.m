clear
close all

load("alex_mask_cheek_n_nose_all_600.mat");
mask1 = double(masks);

load("main_mask_cheek_n_nose_all_600_pilpoly_refinedmediapipe.mat");
mask2 = double(masks);

difference = abs(mask1 - mask2);
numPixelDiff = sum(difference, [2 3]);

maxPixelDiff = max(numPixelDiff);
minPixelDiff = min(numPixelDiff);
avgPixelDiff = mean(numPixelDiff);
stdPixelDiff = std(numPixelDiff);

disp('Max Pixel Diff: ' + string(maxPixelDiff))
disp('Min Pixel Diff: ' + string(minPixelDiff))
disp('Avg Pixel Diff: ' + string(avgPixelDiff))
disp('StDev Pixel Diff: ' + string(stdPixelDiff))

plot(numPixelDiff)
xlabel('Frame Number')
ylabel('# Of Pixels Different')

if true
    figure()
    for i=1:600
        frame = reshape(difference(i,:,:), 480, 640);
        frame = frame(100:380, 100:540);
        frame = frame*255;
    
        RGB = insertText(frame, [50 50], string(i), 'AnchorPoint','LeftBottom');
        RGB = imresize(RGB, 2);
        imshow(RGB)
    end
end
