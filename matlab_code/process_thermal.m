%% Load in baseline measurement
participant = 'patrick';
%load(['./20220428_' participant '_cold_b/' participant '_cold_b.mat']);
load("linearModel.mat")
%load("sk_test_base_alex.mat");
load("Team_baseline_measurements\isiah_base_bfsig.mat");
Depth(:,1) = [];
I_raw(:,1) = [];

I_raw_b = I_raw;
Depth_b = Depth;
I_comp_b = I_raw_b./(Depth_b*linearModel(1)+linearModel(2));

%% Load in second measurement (e.g. exercise recovery period)
%load(['./20220428_' participant '_cold_rec/' participant '_cold_rec.mat']);
%load("sk_test.mat")
load("linearModel.mat")
%load("cold_measurements\isiah_cold_bfsig.mat");
%load("heat_measurements\isiah_heat_bfsig.mat");
%load("lex_measurements\isiah_lex_bfsig.mat");
load("fex_measurements\isiah_fex_bfsig.mat");

Depth(:,1) = [];
I_raw(:,1) = [];
%linearModel(1) = -1;

I_raw_rec = I_raw;
Depth_rec = Depth;
I_comp_rec = I_raw_rec./(Depth_rec*linearModel(1)+linearModel(2));



%% Process waveforms into the different regions

% 1: nose;  2: forehead;   3: nose & cheek  4: left cheek   5: right cheek
Fs = 30;
T=1/Fs;

HRsig_b = I_comp_b(3,:);
HRsig_rec = I_comp_rec(3,:);

I_comp_b_norm = I_comp_b./mean(I_comp_b,2);
I_comp_rec_norm = I_comp_rec./mean(I_comp_b,2);


bc_forehead_b = smooth(-log(I_comp_b_norm(2,:)),20);
bc_nose_b = smooth(-log(I_comp_b_norm(1,:)),20);
bc_lc_b = smooth(-log(I_comp_b_norm(4,:)),20);
bc_rc_b = smooth(-log(I_comp_b_norm(5,:)),20);
bc_lf_b = smooth(-log(I_comp_b_norm(6,:)),20);
bc_pm_b = smooth(-log(I_comp_b_norm(7,:)),20);


bc_forehead_rec = smooth(-log(I_comp_rec_norm(2,:)),20);
bc_nose_rec = smooth(-log(I_comp_rec_norm(1,:)),20);
bc_lc_rec = smooth(-log(I_comp_rec_norm(4,:)),20);
bc_rc_rec = smooth(-log(I_comp_rec_norm(5,:)),20);
bc_lf_rec = smooth(-log(I_comp_rec_norm(6,:)),20);
bc_pm_rec = smooth(-log(I_comp_rec_norm(7,:)),20);


%%
tb = (0:size(I_raw_b,2)-1)*T;
trec = (0:size(I_raw_rec,2)-1)*T;

%% Plot forehead raw intensity vs forehead compensated intensity
figure()
yyaxis left
plot(I_raw_rec(1,:))

yyaxis right
plot(I_comp_rec(1,:))

%% Plot Baseline vs recovery for the various tests

% figure()
% yyaxis left
% hold on
% plot(tb, bc_forehead_b);
% plot(trec+80, bc_forehead_rec);
% yyaxis right
% hold on
% plot(t_HR_b, HR_b)
% plot(t_HR_rec+80, HR_rec)
% ylim([0 150])
% title('Forehead Bloodflow');
% ylabel('Relative Blood Flow (a.u.)');
% xlabel('Time (s)');
% 
figure()
hold on
plot(tb, bc_lf_b);
plot(trec+120, bc_lf_rec);
title('Alex Forehead Bloodflow');
ylabel('Relative Blood Flow (a.u.)');
xlabel('Time (s)');
legend('baseline','recovery');
forehead_test = mean(bc_lf_rec)
forehead_base = mean(bc_lf_b)

figure()
hold on
plot(tb, bc_nose_b);
plot(trec+80, bc_nose_rec);
legend('Baseline','High Intensity','Medium Intensity','Low Intensity');
title('Nose Bloodflow');
ylabel('Relative Blood Flow (a.u.)');
xlabel('Time (s)');
nose_test = mean(bc_nose_rec)
nose_base = mean(bc_nose_b)

figure()
hold on
plot(tb, (bc_lc_b+bc_rc_b)/2);
plot(trec+80, (bc_lc_rec+bc_rc_rec)/2);
legend('Baseline','High Intensity','Medium Intensity','Low Intensity');
title('Cheek Bloodflow');
ylabel('Relative Blood Flow (a.u.)');
xlabel('Time (s)');
cheek_test = mean((bc_lc_rec+bc_rc_rec)/2)
cheek_base = mean((bc_lc_b+bc_rc_b)/2)

% figure(5)
% hold on
% plot(tb, bc_pm_b);
% plot(trec+80, bc_pm_rec);
% legend('Baseline','High Intensity','Medium Intensity','Low Intensity');
% title('Palm Bloodflow');
% ylabel('Relative Blood Flow (a.u.)');
% xlabel('Time (s)');

% figure(6)
% hold on
% plot(trec, bc_forehead_rec);
% plot(trec, bc_nose_rec);
% plot(trec, (bc_lc_rec+bc_rc_rec)/2);
% plot(trec, bc_pm_rec);
% yline(0)
% 
% legend('Forehead', 'Nose', 'Cheek', 'Baseline');
% ylabel('Relative Blood Flow (a.u.)');
% xlabel('Time (s)');
% title('Alex Recovery Blood Concentration');


% figure(7)
% hold on

% plot(trec, bc_nose_rec-bc_forehead_rec);

% ylabel('Relative Blood Flow (a.u.)');
% xlabel('Time (s)');
% title('Heat Stimulus');



%% Fourier Transform
L=900;
step = 30;
step_t = step/Fs;
L_t = L/Fs;
f = Fs*(0:(L/2))/L;
HR_b = [];
HR_rec = [];
RR = [];
j=1;
counter = 1;
while j+L-1 < size(HRsig_b,2)
    spectrum=fft(HRsig_b(j:j+L-1));
    P2 = abs(spectrum./L);
    onesided(j,:) = P2(1:L/2+1);
    onesided(j,2:end-1) = 2*onesided(j,2:end-1);
    f = Fs*(0:(L/2))/L*60;
    f_Filtered_range=f<70|f>200;
    onesided(j,f_Filtered_range)=0;
% HR peak locate
    [pks,loc]=findpeaks(onesided(j,:)) ;
    [maxval,maxindex]=max(pks);
    HR_current = f(loc(maxindex));
    HR_b = [HR_b HR_current];
    
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

%figure()
%plot(f(70:200),onesided(70:200))

j=1;
counter = 1;
while j+L-1 < size(HRsig_rec,2)
    spectrum=fft(HRsig_rec(j:j+L-1));
    P2 = abs(spectrum./L);
    onesided(j,:) = P2(1:L/2+1);
    onesided(j,2:end-1) = 2*onesided(j,2:end-1);
    f = Fs*(0:(L/2))/L*60;
    f_Filtered_range=f<70|f>200;
    onesided(j,f_Filtered_range)=0;
% HR peak locate
    [pks,loc]=findpeaks(onesided(j,:)) ;
    [maxval,maxindex]=max(pks);
    HR_current = f(loc(maxindex));
    HR_rec = [HR_rec HR_current];
    
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

%figure()
%plot(f(70:200),onesided(70:200))

HR_b_med = median(HR_b)
HR_rec_med = median(HR_rec)

t_HR_b = (L_t/2:step_t:((length(HR_b)-1)*step_t+L_t/2));
t_HR_rec = (L_t/2:step_t:((length(HR_rec)-1)*step_t+L_t/2));

figure()
hold on
plot(t_HR_b, HR_b)
plot(t_HR_rec, HR_rec)
legend('baseline', 'Test')
title('Heartrate')
xlabel('Time (seconds)')
ylabel('Heart Rate (bpm)')
HR_test = mean(HR_rec)
HR_base = mean(HR_b)


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
plot(tb,smooth(bc_nose,50));
plot(tb,smooth(bc_forehead,50));
plot(tb,smooth((bc_lc+bc_rc)/2,50));

xlabel('Time (s)')
legend('nose','forehead','cheek')
ylabel('Relative Blood Concentration Change (a.u.)')

subplot(2,1,2)
plot(time_remap_downspl,tp10_ds,'LineWidth',1.5);


figure(9)
yyaxis left
plot(tb,smooth(smooth(int1_comp_b,13)-smooth(int2_comp,13),50))
xlabel('Time (s)')
ylabel('A.U.')

yyaxis right
hold on
plot(t_HR,smooth(HR,3))

figure()
yyaxis left
plot(tb,smooth(smooth(int1_raw,13)-smooth(int2_raw,13),50))
xlabel('Time (s)')
ylabel('A.U.')

yyaxis right
hold on
plot(tb,smooth(smooth(int1_comp_b,13)-smooth(int2_comp,13),50))


% plot(t_HR,smooth(RR,3))

xline(event1_t,'--r',event1)
xline(event2_t,'--r',event2)


legend('Blood volume difference (Nose-Forehead)','Heart Rate')
ylabel('BPM')
xlim([-100 1100]);
ylim([50 100]);

figure()
yyaxis left
plot(tb,smooth(-(bc_lc+bc_rc)/2+bc_forehead,50),'LineWidth',1.5)
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
plot(tb,smooth(-bc_forehead,50),'LineWidth',1.5)
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
plot(tb,smooth(-bc_nose-bc_lc-bc_rc,50),'LineWidth',1.5)
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
plot(tb(1:end),smooth(bc_nose(1:end),50),'LineWidth',1.5)
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
plot(tb(300:end-600),filtered_dbv(300:end-600),'LineWidth',2)

yyaxis right
hold on
plot(HR_time,HR_ref,'--k')
xlim([-100 1100]);

dbv_std = []
dbv_len = length(dbv);
for i = 300:(dbv_len-1500);
    dbv_std(i-299) = std(filtered_dbv(i:i+900));
    
end

