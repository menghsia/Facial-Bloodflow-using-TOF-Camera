function [comp] = depthComp(I_raw,Depth)

% clips will be split into 2 second clips(60 frames) and the a and b values with the minimum correlation will be
% calculated for each 2 second clip 
timeWindow = 2; %seconds
Fs = 30; %frames per second

% make variables for I_comp (to be appended) and i to iterate through
comp = ones([1, size(I_raw,2)]);
i = 1;

%iterate through every clip...so every 60 frames
while (i*60)<length(I_raw)
    b = 0;
    cor = 1;

    % for each clip iterate through different b values with a = 1
    for bi = 0.2:0.01:5
        bI_comp = I_raw(((i-1)*60+1):(i*60))./((Depth(((i-1)*60+1):(i*60)).^(-bi)));
        % find correlation between bI_comp and Depth
        corr_v = corrcoef(bI_comp,Depth(((i-1)*60+1):(i*60)));
        % take absolute value of correlation coefficients
        corr_ = abs(corr_v(1,2));
        
        % if the new correlation coeff is less than the old one reset b
        % value and I_comp
        if corr_ < cor
            cor = corr_;
            best = bI_comp;
            bestB = bi;
        end
    end
    comp(((i-1)*60+1):(i*60)) = best/mean(best);
    i = i+1;
    disp(bestB)
end

% for the remainder
cor = 1;
for bi = 0.1:0.01:5
    bI_comp = I_raw((((i-1)*60)+1):end)./((Depth((((i-1)*60)+1):end).^(-bi)));
    % find correlation between bI_comp and Depth
    corr_v = corrcoef(bI_comp,Depth((((i-1)*60)+1):end));
    % take absolute value of correlation coefficients
    corr_ = abs(corr_v(1,2));
        
    % if the new correlation coeff is less than the old one reset b
    % value and I_comp
    if corr_ < cor
        cor = corr_;
        best = bI_comp;
    end
end
comp((((i-1)*60)+1):end) = best/mean(best); 


