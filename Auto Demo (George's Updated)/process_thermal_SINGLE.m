clear
close all

%% Load in baseline measurement
participant = 'patrick';
load("linearModel.mat")
load("isiah_base_bfsig.mat");

Depth(:,1) = [];
I_raw(:,1) = [];

I_raw_b = I_raw;
Depth_b = Depth;
I_comp_b = I_raw_b./(Depth_b*linearModel(1)+linearModel(2));

%% Process waveforms into the different regions

% 1: nose;  2: forehead;   3: nose & cheek  4: left cheek   5: right cheek
Fs = 30;
T=1/Fs;

HRsig_b = I_comp_b(3,:);

I_comp_b_norm = I_comp_b./mean(I_comp_b,2);

bc_forehead_b = smooth(-log(I_comp_b_norm(2,:)),20);
bc_nose_b = smooth(-log(I_comp_b_norm(1,:)),20);
bc_lc_b = smooth(-log(I_comp_b_norm(4,:)),20);
bc_rc_b = smooth(-log(I_comp_b_norm(5,:)),20);
bc_lf_b = smooth(-log(I_comp_b_norm(6,:)),20);
bc_pm_b = smooth(-log(I_comp_b_norm(7,:)),20);

tb = (0:size(I_raw_b,2)-1)*T;


%% Plot forehead raw intensity vs forehead compensated intensity
figure()
yyaxis left
plot(I_raw_b(1,:))
ylabel('Raw Forehead Intensity')

yyaxis right
plot(I_comp_b(1,:))
ylabel('Compensated Forehead Intensity')

xticks([])

%% Fourier Transform
L=900;
step = 30;
step_t = step/Fs;
L_t = L/Fs;
f = Fs*(0:(L/2))/L;
HR_b = [];

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
        
    j = j+step;
    counter = counter+1;    
end

HR_b_med = median(HR_b)
t_HR_b = (L_t/2:step_t:((length(HR_b)-1)*step_t+L_t/2));

figure()
hold on
plot(t_HR_b, HR_b)
legend('baseline')
title('Heartrate')
xlabel('Time (seconds)')
ylabel('Heart Rate (bpm)')
HR_base = mean(HR_b);


%% Plot Result
figure()
hold on
plot(tb,smooth(bc_nose_b,50));
plot(tb,smooth(bc_forehead_b,50));
plot(tb,smooth((bc_lc_b+bc_rc_b)/2,50));

xlabel('Time (s)')
legend('nose','forehead','cheek')
ylabel('Relative Blood Concentration Change (a.u.)')
