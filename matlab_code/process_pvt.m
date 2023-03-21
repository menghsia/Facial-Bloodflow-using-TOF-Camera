
load(['alex_sleep.mat'])

img = reshape(grayscale,640,480,600);
%%
% participant = 'patrick';
% load(['./20220428_' participant '_cold_b/' participant '_cold_b.mat']);

Depth(:,1:15) = [];
I_raw(:,1:15) = [];
drowsiness = [];
I_comp = I_raw./(Depth*linearModel(1)+linearModel(2));
%%
% 1: nose;  2: forehead;   3: nose & cheek  4: left cheek   5: right cheek
Fs = 30;
T=1/Fs;



I_comp_norm = I_comp./mean(I_comp(:,1:60),2);



bc_forehead = smooth(-log(I_comp_norm(2,:)),20);
bc_nose = smooth(-log(I_comp_norm(1,:)),20);
bc_lc = smooth(-log(I_comp_norm(4,:)),20);
bc_rc = smooth(-log(I_comp_norm(5,:)),20);
bc_lf = smooth(-log(I_comp_norm(6,:)),20);
% bc_pm_b = smooth(-log(I_comp_b_norm(7,:)),20);



t = (0:length(I_raw(1,:))-1)*T;

for i=1:length(elapse)-2

    drowsiness(i) = sum(reactiontime(i:i+2)>1000);
end
plot(elapse(1:end-2),drowsiness);
%% find onset point

index = find(drowsiness==3);
onset = floor(elapse(index(1)));
startpt = onset-100;
endpt = onset+500;
[M, Ist] = min(abs(t-startpt));
[M, Ied] = min(abs(t-endpt));
t_section = (0:Ied-Ist-1)*T;

bc_forehead_section(3,:) = bc_forehead(Ist+1:Ied);
bc_nose_section(3,:) = bc_nose(Ist+1:Ied);

nfsig = bc_nose_section-bc_forehead_section;

% Down sampling
for j=1:(Ied-Ist)/150
    nfsig_dsp(:,j) = mean(nfsig(:,(j-1)*150+1:j*150),2)
end

t_dsp = t_section(75:150:end);

figure()
hold on
plot(t_section, mean(bc_nose_section,1)-mean(bc_forehead_section,1));
xline(100)

figure()
hold on
errorbar(t_dsp, mean(nfsig_dsp,1),std(nfsig_dsp,0,1)/sqrt(size(nfsig_dsp,1)));
xline(100,'-','Onset of sleep')
legend('Differential blood flow (nose - forehead), errorbar = S.E.');
xlabel('Time (s)')
ylabel('Relative Blood Concentration (a.u.)');

%%
figure(1)
subplot(2,1,1)
yyaxis left
plot(t, bc_nose);

yyaxis right
plot(elapse, reactiontime)

subplot(2,1,2)
plot(t,Depth(1,:))


figure(11)
yyaxis left
plot(t, smooth(bc_nose-bc_forehead,60));
ylabel('a.u')

yyaxis right
plot(elapse(1:end-2),drowsiness);
ylim([-0.5 3.5])
xlabel('Time')
ylabel('Drowsiness Level')
legend('Blood Concentration Difference: nose-forehead','Reaction Time')


figure(12)
subplot(2,1,1)
yyaxis left
plot(t, smooth(bc_forehead,60));
yyaxis right
plot(elapse, reactiontime)
subplot(2,1,2)
plot(t,Depth(2,:))

figure(111)
yyaxis left
plot(t, smooth(bc_nose-(bc_lc+bc_rc)/2,60));

yyaxis right
plot(elapse, reactiontime)

figure(1111)
yyaxis left
plot(t, smooth((bc_lc+bc_rc)/2,60));

yyaxis right
plot(elapse, reactiontime)

figure(2)
hold on
plot(t, bc_nose)
plot(t, bc_forehead)
plot(t, bc_lc)
plot(t, bc_rc)
plot(t,bc_lf)
legend('nose','forehead','lc','rc','lf')

figure(3)
hold on
yyaxis left
plot(t,bc_nose);
ylabel('a.u.');
yyaxis right
plot(t,Depth(1,:));
legend('Blood Concentration (nose)','Distance')
xlabel('Time (s)')
ylabel('Distance (mm)')

figure(4)
hold on
yyaxis left
plot(t,bc_forehead);
yyaxis right
plot(t,Depth(2,:));

figure(5)
hold on
yyaxis left
plot(t,bc_lc);
yyaxis right
plot(t,Depth(4,:));


figure(6)
hold on
yyaxis left
plot(t,bc_rc);
yyaxis right
plot(t,Depth(5,:));

figure()
hold on
yyaxis left
plot(t,I_raw(1,:)/I_raw(1,1))
yyaxis right
plot(t,	Depth(1,:))
title('Nose')

figure()
hold on
yyaxis left
plot(t,I_raw(2,:)/I_raw(2,1))
yyaxis right
plot(t,	Depth(2,:))
title('Forehead')

figure()
hold on
yyaxis left
plot(t,I_raw(6,:)/I_raw(6,1))
yyaxis right
plot(t,	Depth(6,:))
title('LF')
figure()
hold on
yyaxis left
plot(t,I_raw(4,:)/I_raw(4,1))
yyaxis right
plot(t,	Depth(4,:))
title('Lc')

figure()
hold on
yyaxis left
plot(t,I_raw(5,:)/I_raw(5,1))
yyaxis right
plot(t,	Depth(5,:))
title('rc')

figure()
hold on
yyaxis left
plot(t,I_raw(1,:)/I_raw(1,1))
yyaxis right
plot(t,I_comp(1,:)/I_comp(1,1))


%% Fourier Transform
L=1800;
step = 90;
step_t = step/Fs;
L_t = L/Fs;
f = Fs*(0:(L/2))/L;
HR = [];
RR = [];
j=1;
counter = 1;
while j+L-1 < size(I_raw(3,:),2)
    spectrum=fft(smooth(I_raw(2,j:j+L-1)));
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

t_HR = (L_t/2:step_t:((length(HR)-1)*step_t+L_t/2));%%


figure()
plot(f,onesided)
xlim([0 220])

%%

j = 7000;
signal = smooth(I_raw(3,j:j+L-1),30);
signal = (signal-mean(signal))/std(signal);
spectrum=fft(signal);
P2 = abs(spectrum./L);
onesided = P2(1:L/2+1);
onesided(2:end-1) = 2*onesided(2:end-1);
f = Fs*(0:(L/2))/L*60;

figure()
plot(f,onesided)
xlim([0 220])
figure()
plot(signal)
% f_Filtered_range=f<70|f>200;
% onesided(f_Filtered_range)=0;
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

