function [comp] = depthComp(I_raw,Depth,timeWindow, Fs)

% clips will be split into the different ROIs and then split into 2 second clips(60 frames) and the a and b values with the minimum correlation will be
% calculated for each 2 second clip for each ROI 

% make matrix for final output
comp = ones(size(I_raw));

% iterate through the different ROIs
for j = 1:1:size(I_raw, 1)

    % make variables for I_comp (to be appended) and i to iterate through
    compj = ones([1, size(I_raw,2)]);
    i = 1;
    
    %iterate through every clip...so every 60 frames
    while (i*(timeWindow*Fs))<length(I_raw(j,:))
        cor = 1;
    
        % for each clip iterate through different b values with a = 1
        for bi = 0.2:0.01:5
            bI_comp = I_raw(j,((i-1)*(timeWindow*Fs)+1):(i*(timeWindow*Fs)))./((Depth(j,((i-1)*(timeWindow*Fs)+1):(i*(timeWindow*Fs))).^(-bi)));
            % find correlation between bI_comp and Depth
            corr_v = corrcoef(bI_comp,Depth(j,((i-1)*(timeWindow*Fs)+1):(i*(timeWindow*Fs))));
            % take absolute value of correlation coefficients
            corr_ = abs(corr_v(1,2));
            
            % if the new correlation coeff is less than the old one reset cor
            % value and I_comp
            if corr_ < cor
                cor = corr_;
                best = bI_comp;
            end
        end
        % normalize data
        compj(((i-1)*(timeWindow*Fs)+1):(i*(timeWindow*Fs))) = best/mean(best);
        i = i+1;
    end
    
    % for the remainder
    cor = 1;
    for bii = 0.1:0.01:5
        bI_comp = I_raw(j,(((i-1)*(timeWindow*Fs))+1):end)./((Depth(j,(((i-1)*(timeWindow*Fs))+1):end).^(-bii)));
        % find correlation between bI_comp and Depth
        corr_v = corrcoef(bI_comp,Depth(j,(((i-1)*(timeWindow*Fs))+1):end));
        % take absolute value of correlation coefficients
        corr_ = abs(corr_v(1,2));
            
        % if the new correlation coeff is less than the old one reset cor
        % value and I_comp
        if corr_ < cor
            cor = corr_;
            best_comp = bI_comp;
        end
    end
    % normalize data
    compj((((i-1)*(timeWindow*Fs))+1):end) = best_comp/mean(best_comp); 
    % append to final output matrix
    comp(j,:) = compj;
end


