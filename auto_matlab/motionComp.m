function [specW, HR, specND, HR_ND] = motionComp(HRsig, depth)
%HRsig is 20 seconds (600 frames), with associated depth data
specW = zeros([1 151]);
specND = zeros([1 151]);
i = 1;

while i <= 301
    % FFT for spectrum, convert to one sided
    spec = fft(HRsig(i:(i+299)));
    P2 = abs(spec/300);
    P1 = P2(1:300/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    % Find top two peaks
    [peaks, ~] = findpeaks(P1);
    peaks = sort(peaks, 'descend');
    A1 = peaks(1);
    A2 = peaks(2);
    
    %Calculate motion score
        score = (var(depth(i:i+299)) - mean(depth(i:i+299)))/(A1/A2);
        score = abs(score);

    %Calculate final spectrum
    specW = specW + P1/score;
    specND = specND + P1;
    i = i + 30;
end

%Calculate final heart rate
f = 30*(0:(300/2))/300*60;
f_Filtered_range=f<40|f>150;
specW(f_Filtered_range)=0;

[peaks,loc]=findpeaks(specW) ;
[maxval,maxindex]=max(peaks);
HR = f(loc(maxindex));

%Calculate final heart rate (No motion comp)
specND(f_Filtered_range)=0;

[peaks,loc]=findpeaks(specND) ;
[maxval,maxindex]=max(peaks);
HR_ND = f(loc(maxindex));

% Plot and display results
figure()
hold on
xlabel('Heart Rate (BPM)')
title('Heart Rate Frequency Spectrum')
xlim([40 150])

yyaxis left
plot(f, specW)
yticks([])

yyaxis right
plot(f, specND)
yticks([])

legend('W/ MC', 'W/O MC')
disp(['Heart rate (With Motion Comp): ' num2str(HR)])
disp(['Heart rate (W/O Motion Comp: ' num2str(HR_ND)])