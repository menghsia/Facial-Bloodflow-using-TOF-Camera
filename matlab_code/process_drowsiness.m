
load(['alex_sleep.mat'])

img = reshape(grayscale,640,480,600);
%%
Depth(:,1) = [];
I_raw(:,1) = [];

event1 = 'event1';
event2 = 'event2';

I_comp = I_raw./(Depth*linearModel(1)+linearModel(2));

% startT = 616; % For drowsy_1
event1_t = 0;   % For drowsy_2
% endT = 927;
event2_t = 945;   % For drowsy_2
Fs = 30;
T=1/Fs;



int1_raw = I_raw(2,:);
int2_raw = I_raw(1,:);
int3_raw = I_raw(3,:);
int4_raw = I_raw(4,:);
int5_raw = I_raw(5,:);



int1_comp = smooth(I_comp(2,:),20);
int2_comp = smooth(I_comp(1,:),20);
int3_comp = smooth(I_comp(3,:),20);
int4_comp = smooth(I_comp(4,:),20);
int5_comp = smooth(I_comp(5,:),20);



int1_comp_norm = int1_comp/mean(int1_comp(1:60));
int2_comp_norm = int2_comp/mean(int2_comp(1:60));
int3_comp_norm = int3_comp/mean(int3_comp(1:60));
int4_comp_norm = int4_comp/mean(int4_comp(1:60));
int5_comp_norm = int5_comp/mean(int5_comp(1:60));

bc_forehead = -log(int1_comp_norm);
bc_nose = -log(int2_comp_norm);
bc_lc = -log(int4_comp_norm);
bc_rc = -log(int5_comp_norm);

% int1 represents intensity on forehead, int2 represents intensity on nose,
% int3 represents intensity on cheek and nose region

dis1 = Depth(2,:);
dis2 = Depth(1,:);
dis3 = Depth(3,:);
dis4 = Depth(4,:);
dis5 = Depth(5,:);


t = (0:length(int1_raw)-1)*T;
%%
figure(1)
subplot(3,1,1)
plot(t,smooth(int1_raw))
subplot(3,1,2)
plot(t,smooth(int1_comp))
subplot(3,1,3)
plot(t,smooth(dis1))

figure(2)
subplot(3,1,1)
plot(t,smooth(int2_raw))
subplot(3,1,2)
plot(t,smooth(int2_comp))
subplot(3,1,3)
plot(t,smooth(dis2))

figure(3)
subplot(3,1,1)
plot(t,smooth(int3_raw))
subplot(3,1,2)
plot(t,smooth(int3_comp))
subplot(3,1,3)
plot(t,smooth(dis3))

figure(4)
subplot(3,1,1)
plot(t,smooth(int4_raw))
subplot(3,1,2)
plot(t,smooth(int4_comp))
subplot(3,1,3)
plot(t,smooth(dis4))



figure(5)
yyaxis left
plot(t,smooth(int1_comp,13));
ylabel('Intensity on Forehead (a.u.)')
yyaxis right
plot(t,smooth(int2_comp,13));
ylabel('Intensity on Nose (a.u.)')
xlabel('Time (s)')
xline(event1_t,'--r',event1)
xline(event2_t,'--r',event2)
legend('Forehead','Nose')


figure(6)
yyaxis left
plot(t,smooth(int2_raw,13))
ylabel('Intensity (a.u.)')
yyaxis right
plot(t,smooth(dis2,13))
ylabel('Distance (m)')
xlabel('Time (s)')
legend('Intensity','Distance')
title('I vs D on nose')

figure(7)
yyaxis left
plot(t,smooth(int1_raw,13))
ylabel('Intensity (a.u.)')
yyaxis right
plot(t,smooth(dis1,13))
ylabel('Distance (m)')
xlabel('Time (s)')
legend('Intensity','Distance')
title('I vs D on forehead')

figure(8)
yyaxis left
plot(t,smooth(int1_comp,13)-smooth(int2_comp,13))

yyaxis right
plot(elapse, reactiontime)
xlabel('Time (s)')
ylabel('A.U.')
legend('Blood flow difference (Nose-Forehead)')

figure(9)
plot(t,smooth(int1_raw,13)-smooth(int2_raw,13))
xlabel('Time (s)')
ylabel('A.U.')
legend('Blood flow difference (Nose-Forehead) no_comp')

figure(10)
plot(t,smooth(dis1,11)-smooth(dis2,11))
xlabel('Time (s)')
ylabel('A.U.')
xline(event1_t,'--r',event1)
xline(event2_t,'--r',event2)
legend('Blood flow difference (Nose-Forehead)')

% close all

%% Fourier Transform
L=900;
step = 90;
step_t = step/Fs;
L_t = L/Fs;
f = Fs*(0:(L/2))/L;
HR = [];
RR = [];
j=1;
counter = 1;
while j+L-1 < size(I_comp(3,:),2)
    spectrum=fft(I_comp(3,j:j+L-1));
    P2 = abs(spectrum./L);
    onesided = P2(1:L/2+1);
    onesided(2:end-1) = 2*onesided(2:end-1);
    f = Fs*(0:(L/2))/L*60;
    f_Filtered_range=f<70|f>200;
    onesided(f_Filtered_range)=0;
% HR peak locate
    [pks,loc]=findpeaks(onesided) ;
    [maxval,maxindex]=max(pks);
    HR_current = f(loc(maxindex));
    HR = [HR HR_current];
    
%     spectrum=fft(RR_sig_concat(2,j:j+L-1));
%     P2 = abs(spectrum./L);
%     onesided = P2(1:L/2+1);
%     onesided(2:end-1) = 2*onesided(2:end-1);
%     f = Fs*(0:(L/2))/L*60;
%     f_Filtered_range=f<5|f>35;
%     onesided(f_Filtered_range)=0;
% HR peak locate
%     [pks,loc]=findpeaks(onesided) ;
%     [maxval,maxindex]=max(pks);
%     RR_current = f(loc(maxindex));
%     
%     RR = [RR RR_current];
%     if counter == 174
%         figure()
%         plot(f,onesided)
%         xlim([0 220])
%     end
        
    j = j+step;
    counter = counter+1;
%     
end

t_HR = (L_t/2:step_t:((length(HR)-1)*step_t+L_t/2));
%%


%% Import RR interval data
% first_skv_time = '03/11/2021 21:30:06'
% RR_interval_filename = '12_10_2021_125958_RR.csv';
% RR_interval_table = readtable(RR_interval_filename,'HeaderLines',4);
% RR_itv_time = table2array(RR_interval_table(:,1));
% RR_itv_time = datenum(char(RR_itv_time),'dd/mm/yyyy HH:MM:SS.FFF');
% input_time = datenum(first_skv_time,'dd/mm/yyyy HH:MM:SS');
% 
% comp_time = 1850/30;
% RR_itv_time = (RR_itv_time-input_time)*24*3600+comp_time;
% 
% RR_interval = table2array(RR_interval_table(:,3));
% figure()
% plot(RR_itv_time,RR_interval)

%% Import HR data
% HR_filename = '03_11_2021_212551_HR.csv';
% HR_table = readtable(HR_filename,'HeaderLines',4);
% HR_time = table2array(HR_table(:,1));
% HR_time = datenum(char(HR_time),'dd/mm/yyyy HH:MM:SS.FFF');
% input_time = datenum(first_skv_time,'dd/mm/yyyy HH:MM:SS');
% HR_time = (HR_time-input_time)*24*3600+comp_time;
% 
% HR_ref = table2array(HR_table(:,3));
% 
% figure()
% plot(HR_time,HR_ref)

%% Calculate HRV

% end_time = 5*floor(RR_itv_time(end)/5);
% HRV_time = (0:5:end_time);
% RR_itv_idx = [];
% HRV = [];
% HRV_bpm = [];
% for i = 1:size(HRV_time,2)
%     [M,RR_itv_idx(i)] = min(abs(HRV_time(i)-RR_itv_time));
% end
% 
% idx = 1;
% while idx+6 <= size(HRV_time,2)
%     
%     HRV(idx) = std(RR_interval(RR_itv_idx(idx):RR_itv_idx(idx+6)));
%     idx = idx+1;
% end
% 
% idx = 1;
% while idx+6 <= size(HRV_time,2)
%     
%     HRV_bpm(idx) = std(60000./RR_interval(RR_itv_idx(idx):RR_itv_idx(idx+6)));
%     idx = idx+1;
% end
% 
% figure()
% plot(HRV_time(4:end-3),HRV_bpm)
%% EEG data extraction
EEG_filename = 'mindMonitor_2022-04-05--19-50-59.csv';
EEG_table = readtable(EEG_filename,'NumHeaderLines',0);
EEG_properties = EEG_table.Properties;

timestamp = table2array(EEG_table(:,1));
timestamp = datenum(timestamp);
timestamp = (timestamp-timestamp(1))*24*3600;

time_remap = (0:size(timestamp)-1)/256;

figure(1)
hold on;
plot(time_remap,EEG_table.RAW_TP9+100);
plot(time_remap,EEG_table.RAW_TP10);
plot(time_remap,EEG_table.RAW_AF7-100);
plot(time_remap,EEG_table.RAW_AF8-200);

% figure(2)
% plot(time_remap,EEG_table.Theta_AF8);
% 
tp9_thetaratio = EEG_table.Theta_TP9-EEG_table.Alpha_TP9;
tp10_thetaratio = EEG_table.Theta_TP10-EEG_table.Alpha_TP10;
af7_thetaratio = EEG_table.Theta_AF7-EEG_table.Alpha_AF7;
af8_thetaratio = EEG_table.Theta_AF8-EEG_table.Alpha_AF8;

time_remap_downspl = downsample(time_remap,10);
tp9_thetaratio_downspl = downsample(tp9_thetaratio,10);
tp10_thetaratio_downspl = downsample(tp10_thetaratio,10);
af7_thetaratio_downspl = downsample(af7_thetaratio,10);
af8_thetaratio_downspl = downsample(af8_thetaratio,10);


tp10_ds = smooth(tp10_thetaratio_downspl,100);
tp9_ds = smooth(tp9_thetaratio_downspl,100);
af7_ds = smooth(af7_thetaratio_downspl,100);
af8_ds = smooth(af8_thetaratio_downspl,100);

% figure(3)
% plot(time_remap_downspl,smooth(tp10_thetaratio_downspl,100));

%%
% num = 213;
% i=256*num+1;
% L=1024;
% Fs=256;
% spectrum=fft(EEG_table.RAW_TP9(i:i+1023)-mean(EEG_table.RAW_TP9(i:i+1023)));
% P2 = abs(spectrum./L);
% onesided = P2(1:L/2+1);
% onesided(2:end-1) = 2*onesided(2:end-1);
% f = Fs*(0:(L/2))/L;
% figure(3)
% hold on
% plot(f,onesided)
% 
% spectrum=fft(EEG_table.RAW_TP10(i:i+1023)-mean(EEG_table.RAW_TP10(i:i+1023)));
% P2 = abs(spectrum./L);
% onesided = P2(1:L/2+1);
% onesided(2:end-1) = 2*onesided(2:end-1);
% f = Fs*(0:(L/2))/L;
% plot(f,onesided)  
% 
% ylim([0 6])


%% Plot Result
dbv = smooth(int1_raw,13)-smooth(int2_raw,13);
filtered_dbv = lowpass(af8_ds,1e-10,Fs,'Steepness',0.999999);
filtered_af8 = lowpass(af8_ds,1e-10,Fs,'Steepness',0.999999);


figure()
subplot(2,1,1)
hold on
plot(t,smooth(bc_nose,50));
plot(t,smooth(bc_forehead,50));
plot(t,smooth((bc_lc+bc_rc)/2,50));

xlabel('Time (s)')
legend('nose','forehead','cheek')
ylabel('Relative Blood Concentration Change (a.u.)')

subplot(2,1,2)
plot(time_remap_downspl,tp10_ds,'LineWidth',1.5);


figure(9)
yyaxis left
plot(t,smooth(smooth(int1_comp,13)-smooth(int2_comp,13),50))
xlabel('Time (s)')
ylabel('A.U.')

yyaxis right
hold on
plot(t_HR,smooth(HR,3))

figure()
yyaxis left
plot(t,smooth(smooth(int1_raw,13)-smooth(int2_raw,13),50))
xlabel('Time (s)')
ylabel('A.U.')

yyaxis right
hold on
plot(t,smooth(smooth(int1_comp,13)-smooth(int2_comp,13),50))


% plot(t_HR,smooth(RR,3))

xline(event1_t,'--r',event1)
xline(event2_t,'--r',event2)


legend('Blood volume difference (Nose-Forehead)','Heart Rate')
ylabel('BPM')
xlim([-100 1100]);
ylim([50 100]);

figure()
yyaxis left
plot(t,smooth(-(bc_lc+bc_rc)/2+bc_forehead,50),'LineWidth',1.5)
xlabel('Time (s)')
ylabel('A.U.')

yyaxis right
hold on
plot(time_remap_downspl,tp10_ds,'LineWidth',1.5);
xline(event1_t,'--r',event1)
xline(event2_t,'--r',event2)
legend('Blood concentration difference (Nose)','\theta wave-\alpha wave')


figure()
yyaxis left
plot(t,smooth(-bc_forehead,50),'LineWidth',1.5)
xlabel('Time (s)')
ylabel('A.U.')

yyaxis right
hold on
plot(time_remap_downspl,tp9_ds,'LineWidth',1.5);
xline(event1_t,'--r',event1)
xline(event2_t,'--r',event2)
legend('Blood concentration difference (forehead)','\theta wave-\alpha wave')

figure()
yyaxis left
plot(t,smooth(-bc_nose-bc_lc-bc_rc,50),'LineWidth',1.5)
xlabel('Time (s)')
ylabel('A.U.')

yyaxis right
hold on
plot(time_remap_downspl,tp10_ds,'LineWidth',1.5);
xline(event1_t,'--r',event1)
xline(event2_t,'--r',event2)
legend('Blood concentration difference (cheek)','\theta wave-\alpha wave')

fhlcrc = (-bc_forehead-bc_lc-bc_rc)/3;

figure()
yyaxis left
plot(t(1:end),smooth(bc_nose(1:end),50),'LineWidth',1.5)
xlabel('Time (s)')
ylabel('A.U.')

yyaxis right
hold on
plot(time_remap_downspl(1:end),tp10_ds(1:end),'LineWidth',1.5);
xline(event1_t,'--r',event1)
xline(event2_t,'--r',event2)
legend('Blood concentration (-cheek-forehead)','\theta wave-\alpha wave')

%% Calculate SNR

dbv = smooth(int1_raw,13)-smooth(int2_raw,13);
diff_dbv = diff(dbv);
filtered_dbv = lowpass(dbv,1e-10,Fs,'Steepness',0.999999);

figure()
hold on
plot(t(300:end-600),filtered_dbv(300:end-600),'LineWidth',2)

yyaxis right
hold on
plot(HR_time,HR_ref,'--k')
xlim([-100 1100]);

dbv_std = []
dbv_len = length(dbv);
for i = 300:(dbv_len-1500);
    dbv_std(i-299) = std(filtered_dbv(i:i+900));
    
end

