clear
close all

%% Load and process data
[tb_b, bc_b, HRsig_b] = processRawData("isiah_heat_bfsig.mat", 'Before');
[tb_rec, bc_rec, HRsig_rec] = processRawData("alex_heat_bfsig.mat", 'After');

%% Plot smoothed blood concentration
f = figure();
subplot(3, 1, 1)
hold on
plot(tb_b,smooth(bc_b(1,:),50));
plot(tb_rec,smooth(bc_rec(1,:),50));
legend('Before', 'After')
title('Forehead')

subplot(3, 1, 2)
hold on
plot(tb_b,smooth(bc_b(2,:),50));
plot(tb_rec,smooth(bc_rec(2,:),50));
legend('Before', 'After')
title('Nose')
ylabel('Relative Blood Concentration Change (a.u.)')

subplot(3, 1, 3)
hold on
plot(tb_b,smooth((bc_b(3,:)+bc_b(4,:))/2,50));
plot(tb_rec,smooth((bc_rec(3,:)+bc_rec(4,:))/2,50));
legend('Before', 'After')
title('Cheek Average')

xlabel('Time (s)')
f.Position(2) = 0;
f.Position(4) = f.Position(4)*2;
%% Get HR Data
[t_HR_b, HR_b] = getHR(HRsig_b);
[t_HR_rec, HR_rec] = getHR(HRsig_rec);

%% Plot HR Data
figure()
hold on
plot(t_HR_b, HR_b)
plot(t_HR_rec, HR_rec)
legend('Before', 'After')
title('Heartrate')
xlabel('Time (seconds)')
ylabel('Heart Rate (bpm)')

