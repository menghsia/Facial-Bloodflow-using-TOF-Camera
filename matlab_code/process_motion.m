% filelist = dir('*.mat');


img = reshape(grayscale,640,480,600);

for i=1:600
    figure(2)
    imshow(img(:,:,i)',[0 600])
    pause(0.02)
end
%%
figure()
subplot(7,1,1)
plot(Depth(2,1:end));
title('Depth')
subplot(7,1,2)
plot(Ang_signal(2,1:end));
title('Angle')
subplot(7,1,3)
plot(I_raw(2,1:end));
title('Intensity')
subplot(7,1,4)
plot(eleCam_signal(2,:))
title('Ele Cam')
subplot(7,1,5)
plot(azCam_signal(2,:))
title('Az Cam')
subplot(7,1,6)
plot(eleObj_signal(2,:))
title('Ele Obj')
subplot(7,1,7)
plot(azObj_signal(2,:))
title('Az Obj')

%%

figure()
plot(smooth(Ang_signal(2,:),30),I_raw(2,:),'.');

figure()
plot(smooth(Ang_signal(2,1:end),30),I_raw(2,:),'.');

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
