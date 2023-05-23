function [t_HR, HR] = getHR(HRsig, L, trial)
%% Prepare Parameters
Fs=30;
step = 30;
step_t = step/Fs;
L_t = L/Fs;
HR = [];

%% Get HR
j=1;
counter = 1;
while j+L-1 < size(HRsig,2)
    spectrum=fft(HRsig(j:j+L-1));
    P2 = abs(spectrum./L);
    onesided(j,:) = P2(1:L/2+1);
    onesided(j,2:end-1) = 2*onesided(j,2:end-1);
    f = Fs*(0:(L/2))/L*60;
    f_Filtered_range=f<40|f>150;
    onesided(j,f_Filtered_range)=0;

% HR peak locate
    [pks,loc]=findpeaks(onesided(j,:)) ;
    [maxval,maxindex]=max(pks);
    HR_current = f(loc(maxindex));
    HR = [HR HR_current];
        
    j = j+step;
    counter = counter+1;    
end

t_HR = (L_t/2:step_t:((length(HR)-1)*step_t+L_t/2));
end

