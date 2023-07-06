clear
close all

%% Load and process data
%filename = "../skvs/mat/auto_bfsig.mat";
filename = "./Test Data/L_5-25_5.mat";

[tb, bc, HRsig, HRsigRaw, I_comp, Depth] = processRawData(filename);

%% Plot smoothed blood concentration
figure()
hold on
plot(tb,smooth(bc(2,:),50));
plot(tb,smooth(bc(1,:),50));
plot(tb, smooth((bc(3,:)+bc(4,:))/2,50));

xlabel('Time (s)')
legend('Nose','Forehead','Cheek Average')
ylabel('Relative Blood Concentration Change (a.u.)')

%% Get HR Data
[t_HR, HR] = getHR(HRsig, 900);

%% Plot HR Data
figure()
plot(t_HR, HR)
title('Heartrate')
xlabel('Time (seconds)')
ylabel('Heart Rate (bpm)')

hold on
[t_HR, HR] = getHR(HRsigRaw, 900);
plot(t_HR, HR, '--')

legend('comp', 'raw')

%% Calculate Heart Rate (Motion Score)
motionComp(HRsig(1:600), Depth(3,1:600));
motionComp(HRsig(601:1200), Depth(3,601:1200));
motionComp(HRsig(1201:1800), Depth(3,1201:1800));