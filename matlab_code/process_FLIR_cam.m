I = imread('IR_1242.jpg');

%Change these values based on bar in picture
bottom_val = 22;
top_val = 36;
%

imshow(I) % Shows image
roi = drawrectangle;

Icropped = imcrop(I,roi.Position);
%imshow(Icropped) % Shows cropped section

for x = 1:1:size(Icropped,1)
    for y = 1:1:size(Icropped,2)
        grey_value(x+y-1)=Icropped(x,y,1);
    end
end
mean_value = mean(grey_value);

mean_value_ratio = mean_value / 255;
mean_temp = mean_value_ratio*(top_val-bottom_val) + bottom_val