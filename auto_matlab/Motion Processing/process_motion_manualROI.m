clear
close all

load('sk_automotive_20221003_164605.skv.mat')

z_frames = reshape(z_value,[640,480,600]);
int_frames = reshape(grayscale,[640,480,600]);
x_frames = reshape(x_value,[640,480,600]);
y_frames = reshape(y_value,[640,480,600]);
%%
frame = uint8(double(int_frames(:,:,1))*255/2000);


figure()
imshow(frame);
roirect1 = drawrectangle;

roirect2 = drawrectangle;

objectImage = insertShape(frame,'Rectangle',roirect1.Position,'Color','red');

points = detectMinEigenFeatures(frame,'ROI',roirect1.Position,'MinQuality',0.005);
pointImage = insertMarker(frame,points.Location,'+','Color','white');
figure;
imshow(pointImage);
title('Detected interest points');

tracker = vision.PointTracker('NumPyramidLevels', 2, 'MaxBidirectionalError',4, 'BlockSize', [21 21] );
initialize(tracker,points.Location,frame);

oldPoints = points.Location;

bboxPoints = bbox2points(roirect2.Position);

mask = poly2mask(double(bboxPoints(:,1)),double(bboxPoints(:,2)),640,480);

I_signal = [];
D_signal = [];
I_signal(1) = mean(nonzeros(double(int_frames(:,:,1)).*mask));
D_signal(1) = mean(nonzeros(double(y_frames(:,:,1)).*mask));

%%

for i=2:size(int_frames,3);
    frame=uint8(double(int_frames(:,:,i))*255/2000);
    [points,validity] = step(tracker,frame);
    visiblePoints = points(validity,:);
    oldInliers = oldPoints(validity,:);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points

        % Estimate the geometric transformation between the old points and
        % the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 2);

        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
        mask = poly2mask(double(bboxPoints(:,1)),double(bboxPoints(:,2)),640,480);
        
        I_signal(i) = mean(nonzeros(double(int_frames(:,:,i)).*mask));
        D_signal(i) = mean(nonzeros(double(y_frames(:,:,i)).*mask));
        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);
        frame = insertShape(frame, 'Polygon', bboxPolygon, ...
            'LineWidth', 2);
        
        % Display tracked points
        frame = insertMarker(frame, visiblePoints, '+', ...
            'Color', 'white');       
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(tracker, oldPoints);
    end
    
    imshow(frame);
end

%% 

figure(1)
funct = I_signal.*D_signal.^1.6;

plot(funct/max(funct));

figure(2)
plot(I_signal,D_signal)
xlabel('Intensity');
ylabel('Distance');
%%
I_crop = I_signal(1:575);
D_crop = D_signal(1:575);

largeInx = D_signal>281;
D_281_p = nonzeros(D_signal.*largeInx);
I_281_p = nonzeros(I_signal.*largeInx);

figure(3)
plot(I_281_p,D_281_p)
xlabel('Intensity');
ylabel('Distance');

r=1;
linearModel = polyfit(D_281_p,I_281_p,1);

I_comp = I_281_p./(D_281_p*linearModel(1)+linearModel(2));

figure()
plot(I_comp);

figure()
hold on
plot(D_crop,I_crop);
plot(D_281_p,I_281_p);
plot(D_281_p,1.09*I_281_p);
xlabel('Distance');
ylabel('Intensity');
legend('Sample 1','Sample 2','r * Sample 2')