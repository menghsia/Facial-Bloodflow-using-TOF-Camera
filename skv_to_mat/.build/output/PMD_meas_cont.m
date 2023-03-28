clc
clear
close all
% retrieve royale version information
number_of_frame=200;
royaleVersion = royale.getVersion();
fprintf('* royale version: %s\n',royaleVersion);

% the camera manager will query for a connected camera
manager = royale.CameraManager();
camlist = manager.getConnectedCameraList();

fprintf('* Cameras found: %d\n',numel(camlist));
cellfun(@(cameraId)...
    fprintf('    %s\n',cameraId),...
    camlist);

if (~isempty(camlist))
    % this represents the main camera device object
    cameraDevice = manager.createCamera(camlist{1});
else
    error(['Please make sure that a supported camera is plugged in, all drivers are ',...
        'installed, and you have proper USB permission']);
end

% the camera device is now available and CameraManager can be deallocated here
delete(manager);

% IMPORTANT: call the initialize method before working with the camera device
cameraDevice.initialize();

% display some information about the connected camera
fprintf('====================================\n');
fprintf('        Camera information\n');
fprintf('====================================\n');
fprintf('Id:              %s\n',cameraDevice.getId());
fprintf('Type:            %s\n',cameraDevice.getCameraName());
fprintf('Width:           %u\n',cameraDevice.getMaxSensorWidth());
fprintf('Height:          %u\n',cameraDevice.getMaxSensorHeight());

% retrieve valid use cases
UseCases=cameraDevice.getUseCases();
fprintf('Use cases: %d\n',numel(UseCases));
fprintf('    %s\n',UseCases{:});
fprintf('====================================\n');

if (numel(UseCases) == 0)
    error('No use case available');
end
    
% % set use case
% UseCase=UseCases{1};

% set use case interactively
UseCaseSelection=listdlg(...
    'Name','Operation Mode',...
    'PromptString','Choose operation mode:',...
    'ListString',UseCases,...
    'SelectionMode','single',...
    'ListSize',[200,200]);
if isempty(UseCaseSelection)
    return;
end
UseCase=UseCases{UseCaseSelection};

cameraDevice.setUseCase(UseCase);
pause(3)
% preview camera
fprintf('* Starting preview. Close figure to exit...\n');
for jkl=0:9
    % start capture mode
    cameraDevice.startCapture();

    % % change the exposure time (limited by the used operation mode [microseconds]
    % fprintf('* Changing exposure time to 200 microseconds...\n');
    % cameraDevice.setExposureTime(200);

    % initialize preview figure
    hFig=figure('Name',...
        ['Preview: ',cameraDevice.getId(),' @ ', UseCase],...
        'IntegerHandle','off','NumberTitle','off');
    colormap(jet(256));
    TID = tic();
    last_toc = toc(TID);
    iFrame = 0;
    grayscale=zeros(171,224,number_of_frame,'uint16');
    distance=zeros(171,224,number_of_frame);
    while (ishandle(hFig))&&iFrame<=number_of_frame
        % retrieve data from camera
        iFrame = iFrame + 1;
        data = cameraDevice.getData();
        grayscale(:,:,iFrame)=data.grayValue;
        distance(:,:,iFrame)=data.z;
        x_value(:,:,iFrame)=data.x;
        y_value(:,:,iFrame)=data.y;
        if (mod(iFrame,10) == 0)
            this_toc=toc(TID);
            fprintf('FPS = %.2f\n',10/(this_toc-last_toc));
            last_toc=this_toc;
        end

        %%% notice: figures are slow.
        %%% For higher FPS (e.g. 45), do not display every frame.
        %%% e.g. by doing here:
        % if (mod(iFrame,5) ~= 0);continue;end;

        % visualize data
       % set(0,'CurrentFigure',hFig);

       % subplot(2,3,1);
       % my_image(data.x,'x');

       % subplot(2,3,2);
       % my_image(data.y,'y');

        %subplot(2,3,3);
        %my_image(data.z,'z');

       % subplot(2,3,4);
       % my_image(data.grayValue,'grayValue');

       % subplot(2,3,5);
        %my_image(data.noise,'noise');

        %subplot(2,3,6);
        %my_image(data.depthConfidence,'depthConfidence');

        %drawnow;
    end

    % stop capture mode
    fprintf('* Stopping capture mode...\n');
    cameraDevice.stopCapture();

    fprintf('* ...done!\n');
    close all

    for i =1:number_of_frame
        grayscale_for_plot(:,:,i)=(grayscale(:,:,i));
    end
    save(sprintf('shawn_different_locations_%f.mat',jkl),'grayscale','distance','number_of_frame','x_value','y_value')
end
