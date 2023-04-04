function [tb, bc, HRsig, x] = processRawData(data, dataTitle)
%% Load in baseline measurement
load(data)

Depth(:,1) = [];
I_raw(:,1) = [];

for i=size(Depth,2):-1:1
    if Depth(1,i) == 0
        Depth(:,i) = [];
    end
    if I_raw(1,i) == 0
    I_raw(:,i) = [];
    end
end

I_comp = I_raw./(Depth*-0.766825576784889+591.153575311779);

%% Process waveforms into the different regions

% 1: nose;  2: forehead;   3: nose & cheek  4: left cheek   5: right cheek
Fs = 30;
T=1/Fs;

HRsig = I_comp(3,:);

I_comp_norm = I_comp./mean(I_comp,2);

bc_forehead = smooth(-log(I_comp_norm(2,:)),20);
bc_nose = smooth(-log(I_comp_norm(1,:)),20);
bc_lc = smooth(-log(I_comp_norm(4,:)),20);
bc_rc = smooth(-log(I_comp_norm(5,:)),20);
bc_lf = smooth(-log(I_comp_norm(6,:)),20);

bc = [bc_forehead'; bc_nose'; bc_lc'; bc_rc'; bc_lf'];
tb = (0:size(I_raw,2)-1)*T;

%% Plot Raw and Compensated Data
figure()
yyaxis left
plot(I_raw(1,:))
ylabel('Raw Intensity')

yyaxis right
plot(I_comp(1,:))
ylabel('Compensated Intensity')
x = I_comp;
if exist('dataTitle', 'var')
    title(['Forehead Signal Intensity: ', dataTitle])
else
    title('Forehead Signal Intensity')
end

xticks([])
end

