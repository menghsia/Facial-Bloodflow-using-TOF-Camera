function [tb, bc, HRsig, HRsigRaw, I_comp_norm] = processRawData(data, dataTitle)
%% Load in baseline measurement
load(data)

Depth(:,1) = [];
I_raw(:,1) = [];

%% Remove extraneous zeros
for i=size(Depth,2):-1:1
    if Depth(1,i) == 0
        Depth(:,i) = [];
        I_raw(:,i) = [];
    else
        break
    end
end

%% Compensate for movement
%I_comp = depthComp(I_raw,Depth);

I_comp_norm = [depthComp(I_raw(1,:),Depth(1,:));depthComp(I_raw(2,:),Depth(2,:)); depthComp(I_raw(3,:),Depth(3,:));depthComp(I_raw(4,:),Depth(4,:));depthComp(I_raw(5,:),Depth(5,:))];
%trim vector
%I_comp_norm = I_comp_norm(:,floor(size(I_comp_norm, 2)*0.25):floor(size(I_comp_norm, 2)*0.75))
%% Process waveforms into the different regions

% 1: nose;  2: forehead;   3: nose & cheek  4: left cheek   5: right cheek
Fs = 30;
T=1/Fs;

HRsig = I_comp_norm(3,:);
HRsigRaw = I_raw(3,:);

%I_comp_norm = I_comp./mean(I_comp,2);

bc_forehead = smooth(-log(I_comp_norm(2,:)),20);
bc_nose = smooth(-log(I_comp_norm(1,:)),20);
bc_lc = smooth(-log(I_comp_norm(4,:)),20);
bc_rc = smooth(-log(I_comp_norm(5,:)),20);
%bc_lf = smooth(-log(I_comp_norm(6,:)),20);

bc = [bc_forehead'; bc_nose'; bc_lc'; bc_rc'];%; bc_lf'];
tb = (0:size(I_raw,2)-1)*T;

%% Plot Raw and Compensated Data
figure()
subplot(2,1,1)
plot(I_raw(1,:))
ylabel('Raw Intensity')

subplot(2,1,2)
plot(I_comp_norm(1,:))
ylabel('Compensated Intensity')

subplot(2,1,1)
if exist('dataTitle', 'var')
    title(['Forehead Signal Intensity: ', dataTitle])
else
    title('Forehead Signal Intensity')
end

xticks([])
end


