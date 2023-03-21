filelist = dir('*.mat');

Depth(1,:)=[]
I_raw(1,:)=[]
for i=1:4;
    
    load(filelist(i).name)
    
    z_concat = [z_concat distance];
    int_concat = [int_concat grayscale];
    x_concat = [x_concat x_value];
    y_concat = [y_concat y_value];
    
end

z_frames = reshape(z_concat,[640,480,size(z_concat,2)]);
int_frames = reshape(int_concat,[640,480,size(int_concat,2)]);
x_frames = reshape(x_concat,[640,480,size(int_concat,2)]);
y_frames = reshape(y_concat,[640,480,size(int_concat,2)]);
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

I_crop3 = I_signal(1:2300);
D_crop3 = D_signal(1:2300);

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
