clear
close all

load('alex_mask_cheek_n_nose_all_600.mat')

for i=1:600
    mask = reshape(masks(i,:,:), 480, 640);
    mask = mask(100:380, 100:540)*255;

    RGB = insertText(mask, [50 50], string(i), 'AnchorPoint','LeftBottom');
    RGB = imresize(RGB, 2);
    imshow(RGB)

end
