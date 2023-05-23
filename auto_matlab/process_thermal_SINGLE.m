clear
close all

%% Load and process data
%[tb, bc, HRsig, HRsigRaw, I_comp_norm] = processRawData("../skvs/mat/auto_bfsig.mat");
[tb, bc, HRsig, HRsigRaw] = processRawData("lauren_5-23.mat");

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