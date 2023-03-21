clear all
close all


%%
load('linearModel.mat')
for pid=2:6
    for i=1:4
        foldername = ['t' num2str(i) '_p' num2str(pid)]
        load(['D:\data\facial_blood\20220212_intoxication\' foldername '\' foldername '_anglesignal.mat'])


        Depth_all{i,pid-1} = Depth;
        % Compensation using linear model
        I_comp_all{i,pid-1} = I_raw./(Depth*linearModel(1)+linearModel(2));

        I_raw_all{i,pid-1} = I_raw;

        I_raw_norm_all{i,pid} = I_raw./mean(I_raw_all{1,pid},2);        
        I_comp_norm_all{i,pid} = I_comp_all{i,pid}./mean(I_comp_all{1,pid},2);

    end
    % 


    I_raw_avg = zeros(4,5);


    for i=1:4
        for j=1:5
            I_raw_norm_avg(i,j) = mean(I_raw_norm_all{i,pid}(j,:));
            I_comp_norm_avg(i,j) = mean(I_comp_norm_all{i,pid}(j,:));


        end
    end

    
    
    bc_forehead(:,pid) = -log(I_comp_norm_avg(:,2));
    bc_nose(:,pid) = -log(I_comp_norm_avg(:,1));
    bc_lc(:,pid) = -log(I_comp_norm_avg(:,4));
    bc_rc(:,pid) = -log(I_comp_norm_avg(:,5));
    
    bc_raw_forehead(:,pid) = -log(I_raw_norm_avg(:,2));
    bc_raw_nose(:,pid) = -log(I_raw_norm_avg(:,1));
    bc_raw_lc(:,pid) = -log(I_raw_norm_avg(:,4));
    bc_raw_rc(:,pid) = -log(I_raw_norm_avg(:,5));
 
end
%%
BAC6 = [0 0.04 0.07 0.112];
BAC5 = [0 0.018 0.045 0.064];
BAC4 = [0 0.014 0.027 0.043];
BAC3 = [0 0.017 0.056 0.086];
BAC2 = [0 0.054 0.109 0.132];
BAC1 = [0 0.033 0.059 0.092];

T1 = [0 20 49 87];
T2 = [0 20 50 87];
T3 = [0 20 50 88];
T4 = [0 20 50 87];
T5 = [0 20 50 86];
T6 = [0 20 50 93];

T_alc = horzcat(T2,T3,T4,T5,T6)/60;
weight = [106 75 108 123 88 79];
r = horzcat(ones(1,3)*0.55,ones(1,16)*0.68);
bc_forehead_weight = bc_forehead./weight;

figure()
hold on
plot(BAC1,bc_forehead(:,1),'-o');
plot(BAC2,bc_forehead(:,2),'-o');
plot(BAC3,bc_forehead(:,3),'-o');
plot(BAC4,bc_forehead(:,4),'-o');
plot(BAC5,bc_forehead(:,5),'-o');
plot(BAC6,bc_forehead(:,6),'-o');
legend('1','2','3','4','5','6');

figure()
hold on
plot(BAC1,(bc_lc(:,1)+bc_rc(:,1))/2,'-o');
plot(BAC2,(bc_lc(:,2)+bc_rc(:,2))/2,'-o');
plot(BAC3,(bc_lc(:,3)+bc_rc(:,3))/2,'-o');
plot(BAC4,(bc_lc(:,4)+bc_rc(:,4))/2,'-o');
plot(BAC5,(bc_lc(:,5)+bc_rc(:,5))/2,'-o');
plot(BAC6,(bc_lc(:,6)+bc_rc(:,6))/2,'-o');
legend('1','2','3','4','5','6');

figure()
hold on
plot(BAC1,bc_nose(:,1),'-o');
plot(BAC2,bc_nose(:,2),'-o');
plot(BAC3,bc_nose(:,3),'-o');
plot(BAC4,bc_nose(:,4),'-o');
plot(BAC5,bc_nose(:,5),'-o');
plot(BAC6,bc_nose(:,6),'-o');
legend('1','2','3','4','5','6');

figure()
hold on
plot(BAC1,bc_forehead(:,1)-(bc_lc(:,1)+bc_rc(:,1))/2,'-o');
plot(BAC2,bc_forehead(:,2)-(bc_lc(:,2)+bc_rc(:,2))/2,'-o');
plot(BAC3,bc_forehead(:,3)-(bc_lc(:,3)+bc_rc(:,3))/2,'-o');
plot(BAC4,bc_forehead(:,4)-(bc_lc(:,4)+bc_rc(:,4))/2,'-o');
plot(BAC5,bc_forehead(:,5)-(bc_lc(:,5)+bc_rc(:,5))/2,'-o');
plot(BAC6,bc_forehead(:,6)-(bc_lc(:,6)+bc_rc(:,6))/2,'-o');
legend('1','2','3','4','5','6');

figure()
hold on
plot(BAC1,bc_forehead(:,1)+bc_nose(:,1),'-o');
plot(BAC2,bc_forehead(:,2)+bc_nose(:,1),'-o');
plot(BAC3,bc_forehead(:,3)+bc_nose(:,1),'-o');
plot(BAC4,bc_forehead(:,4)+bc_nose(:,1),'-o');
plot(BAC5,bc_forehead(:,5)+bc_nose(:,1),'-o');
plot(BAC6,bc_forehead(:,6)+bc_nose(:,1),'-o');
legend('1','2','3','4','5','6');

figure()
hold on
plot(BAC1,bc_forehead_weight(:,1),'o');
plot(BAC2,bc_forehead_weight(:,2),'o');
plot(BAC3,bc_forehead_weight(:,3),'o');
plot(BAC4,bc_forehead_weight(:,4),'o');
plot(BAC5,bc_forehead_weight(:,5),'o');
plot(BAC6,bc_forehead_weight(:,6),'o');
legend('1','2','3','4','5','6');

BAC_all = horzcat(BAC2,BAC3,BAC4,BAC5,BAC6);
bc_forehead_weight_all = vertcat(bc_forehead_weight(:,2), ...
bc_forehead_weight(:,3),bc_forehead_weight(:,4),bc_forehead_weight(:,5),bc_forehead_weight(:,6));

bc_forehead_all = vertcat(bc_forehead(:,2), ...
bc_forehead(:,3),bc_forehead(:,4),bc_forehead(:,5),bc_forehead(:,6));

bc_lc_all = vertcat(bc_lc(:,2), ...
bc_lc(:,3),bc_lc(:,4),bc_lc(:,5),bc_lc(:,6));

bc_rc_all = vertcat(bc_rc(:,2), ...
bc_rc(:,3),bc_rc(:,4),bc_rc(:,5),bc_rc(:,6));


BAC_metacrct = (BAC_all+T_alc*0.016);

for j=1:5
    BAC_correct((j-1)*4+1) = BAC_all((j-1)*4+1)*weight(j+1);
    BAC_correct((j-1)*4+2) = BAC_all((j-1)*4+2)*weight(j+1);
    BAC_correct((j-1)*4+3) = BAC_all((j-1)*4+3)*weight(j+1);
    BAC_correct((j-1)*4+4) = BAC_all((j-1)*4+4)*weight(j+1);

    BAC_meta_wt_crct((j-1)*4+1) = BAC_metacrct((j-1)*4+1)*weight(j+1);
    BAC_meta_wt_crct((j-1)*4+2) = BAC_metacrct((j-1)*4+2)*weight(j+1);
    BAC_meta_wt_crct((j-1)*4+3) = BAC_metacrct((j-1)*4+3)*weight(j+1);
    BAC_meta_wt_crct((j-1)*4+4) = BAC_metacrct((j-1)*4+4)*weight(j+1);
end
    

plot(BAC_correct, bc_forehead_all,'o');





figure()
yyaxis left
plot(BAC_all, bc_forehead_weight_all,'o');
yyaxis right
plot(BAC_all, bc_forehead_all,'o');
legend('BC/Weight','BC Only')

ft = fittype({'x'})



[fit1,gof1]=fit(BAC_all',bc_forehead_all,ft,'Robust','Bisquare')
figure()
plot(fit1,'r-',BAC_all,bc_forehead_all,'o');
xlabel('BAC (%)');
ylabel('Blood Concentration');
text(0.08,0,['R^2 = ' num2str(gof1.rsquare)]);

[fit2,gof2]=fit(BAC_all',bc_forehead_weight_all,ft,'Robust','Bisquare')
figure()
plot(fit2,'r-',BAC_all,bc_forehead_weight_all,'o');
xlabel('BAC (%)');
ylabel('Blood Concentration/Weight');
text(0.08,0,['R^2 = ' num2str(gof2.rsquare)]);

[fit3,gof3]=fit(BAC_correct',bc_forehead_all,ft,'Robust','Bisquare')
figure()
plot(fit3,'r-',BAC_correct,bc_forehead_all,'o');
xlabel('BAC * weight');
ylabel('Blood Concentration');
text(6,0,['R^2 = ' num2str(gof3.rsquare)]);

[fit4,gof4]=fit((BAC_correct.*r)',bc_forehead_all,ft,'Robust','Bisquare')
figure()
plot(fit4,'r-',BAC_correct.*r,bc_forehead_all,'o');
xlabel('BAC * weight * r');
ylabel('Blood Concentration');
text(4,0,['R^2 = ' num2str(gof4.rsquare)]);


[fit5,gof5]=fit((BAC_meta_wt_crct.*r)',bc_forehead_all,ft,'Robust','Bisquare')
figure()
plot(fit5,'r-',BAC_meta_wt_crct.*r,bc_forehead_all,'o');
xlabel('(BAC+metabolic rate correction) * weight * r');
ylabel('Blood Concentration');
ylim([-0.01 0.1])
% text(5,0,['R^2 = 0.84' num2str(gof5.rsquare)]);
text(5,0,['R^2 = 0.84']);


[fit6,gof6]=fit((BAC_meta_wt_crct.*r)',bc_forehead_all-(bc_lc_all+bc_rc_all)/2,ft,'Robust','Bisquare')
figure()
plot(fit6,'r-',BAC_meta_wt_crct.*r,bc_forehead_all-(bc_lc_all+bc_rc_all)/2,'o');
xlabel('(BAC+metabolic rate correction) * weight * r');
ylabel('Blood Concentration (forehead - cheek)');
text(5,0,['R^2 = ' num2str(gof6.rsquare)]);





%%
plotf1 = bc_forehead;
plotf2 = bc_forehead - bc_lc/2 - bc_rc/2;
plotf3 = (bc_forehead+bc_nose+bc_lc+bc_rc)/4;
plotf4 = (bc_forehead+bc_nose-bc_lc-bc_rc)/2;

plotf1_raw = bc_raw_forehead;

figure()
hold on
errorbar((0:3),mean(plotf1,2),std(plotf1,0,2)/sqrt(6),'LineWidth',1.5);
errorbar((0:3),mean(plotf1_raw,2),std(plotf1_raw,0,2)/sqrt(6),'LineWidth',1.5);

xlabel('# of drinks');
ylabel('Blood Concentration Change (a.u.) + SE');
title('Forehead');
legend('Compensated Signal','Uncompensated Signal');


for i=1:4
    depth4_avg(i) = mean(Depth_all{i,4}(2,:))
end

figure()

colororder([0 0.4470 0.7410; 0.4660 0.6740 0.1880])
hold on
yyaxis left
plot((0:3),plotf1(:,4),'LineWidth',1.5);
plot((0:3),plotf1_raw(:,4),'LineWidth',1.5,'Color',[0.8500 0.3250 0.0980]);
ylabel('Blood Concentration Change (a.u.)');

yyaxis right
plot((0:3),depth4_avg,'o','LineWidth',1.5);
legend('Compensated Signal','Uncompensated Signal','Averaged Depth')
ylabel('Distance (mm)')
xlabel('# of drinks');
title('Participant 4, Forehead');


figure()
hold on
plot((0:3),mean(plotf1,2),'LineWidth',1.5);
for i=1:6
    plot((0:3),bc_forehead(:,i),'o','LineWidth',1.5);
end
xlabel('# of drinks');
ylabel('Blood Concentration Change (a.u.) + SE');
title('Forehead');

figure()
hold on
errorbar((0:3),mean(plotf2,2),std(plotf2,0,2)/sqrt(6),'LineWidth',1.5);
xlabel('# of drinks');
ylabel('Blood Concentration Change (a.u.) + SE');
title('Forehead - Cheek');
figure()
hold on
errorbar((0:3),mean(plotf3,2),std(plotf3,0,2)/sqrt(6),'LineWidth',1.5);
xlabel('# of drinks');
ylabel('Blood Concentration Change (a.u.) + SE');
title('All Region');
figure()
hold on
errorbar((0:3),mean(plotf4,2),std(plotf4,0,2)/sqrt(6),'LineWidth',1.5);
xlabel('# of drinks');
ylabel('Blood Concentration Change (a.u.) + SE');
title('Forehead + Nose - Cheek');

figure()
hold on
errorbar((0:3),mean(plotf1,2),std(plotf1,0,2)/sqrt(6),'LineWidth',1.5,'CapSize',20);
errorbar((0:3),mean(plotf2,2),std(plotf2,0,2)/sqrt(6),'LineWidth',1.5,'CapSize',20);
% errorbar((0:3),mean(plotf3,2),std(plotf3,0,2)/sqrt(6),'LineWidth',1.5,'CapSize',20);
errorbar((0:3),mean(plotf4,2),std(plotf4,0,2)/sqrt(6),'LineWidth',1.5,'CapSize',20);
xlabel('# of drinks');
ylabel('Blood Concentration Change (a.u.) + SE');
legend('Forehead','Forehead - Cheek','Forehead+Nose-Cheek');

figure()
hold on
plot((0:3),mean(plotf1,2),'-o','LineWidth',1.5);
plot((0:3),mean(plotf2,2),'-o','LineWidth',1.5);
plot((0:3),mean(plotf4,2),'-o','LineWidth',1.5);
xlabel('# of drinks');
ylabel('Blood Concentration Change (a.u.) + SE');
legend('Forehead','Forehead - Cheek','Forehead+Nose-Cheek');

% figure()
% hold on
% errorbar((0:3),mean(bc_forehead(:,2:6),2),std(bc_forehead(:,2:6),0,2)/sqrt(5));
% errorbar((0:3),mean(bc_forehead(:,2:6),2),std(bc_forehead(:,2:6),0,2)/sqrt(1));
% 
% figure()
% hold on
% for i=1:6
%     plot((0:3),bc_forehead(:,i));
% end


%%

BAC6 = [0 0.04 0.07 0.112];
BAC5 = [0 0.018 0.045 0.064];
BAC4 = [0 0.014 0.027 0.043];
BAC3 = [0 0.017 0.056 0.086];
BAC2 = [0 0.054 0.109 0.132];
BAC1 = [0 0.033 0.059 0.092];

BAC = BAC1 % Choose your BAC:

figure()
hold on
plot(BAC5, bc_forehead(:,1));

figure()
hold on
plot(BAC,-log(I_comp_norm_avg(:,2)),'-o');

xlabel('BAC (%)');
ylabel('Blood Concentration Change (a.u.)');
legend('Forehead');
title('Participant #');

figure()
hold on
plot(BAC,-log(I_comp_norm_avg(:,2)),'-o');
plot(BAC,-log(I_comp_norm_avg(:,4)),'-o');
plot(BAC,-log(I_comp_norm_avg(:,5)),'-o');
plot(BAC,-log(I_comp_norm_avg(:,1)),'-o');

xlabel('BAC (%)');
ylabel('Blood Concentration Change (a.u.)');
legend('Forehead','Left Cheek','Right Cheek','Nose');
title('Participant #');


figure()
hold on
plot(BAC,(log(I_comp_norm_avg(:,4))+log(I_comp_norm_avg(:,5)))/2-log(I_comp_norm_avg(:,2)),'-o');

xlabel('BAC (%)');
ylabel('Blood Concentration Change (a.u.)');
legend('Forehead - Cheek');
title('Participant #');

%%

figure()

yyaxis left
hold on
plot([1:length(I_raw_all{1}(1,:))],I_raw_all{1}(1,:))
plot([2001:2000+length(I_raw_all{2}(1,:))],I_raw_all{2}(1,:))
plot([4001:4000+length(I_raw_all{3}(1,:))],I_raw_all{3}(1,:))
plot([6001:6000+length(I_raw_all{4}(1,:))],I_raw_all{4}(1,:))
ylabel('Intensity (a.u.)')
yyaxis right
hold on
plot([1:length(I_raw_all{1}(1,:))],Depth_all{1}(1,:))
plot([2001:2000+length(I_raw_all{2}(1,:))],Depth_all{2}(1,:))
plot([4001:4000+length(I_raw_all{3}(1,:))],Depth_all{3}(1,:))
plot([6001:6000+length(I_raw_all{4}(1,:))],Depth_all{4}(1,:))
ylabel('Distance (m)')
title('Nose')

figure()

yyaxis left
hold on
plot([1:length(I_raw_all{1}(2,:))],I_raw_all{1}(2,:))
plot([2001:2000+length(I_raw_all{2}(2,:))],I_raw_all{2}(2,:))
plot([4001:4000+length(I_raw_all{3}(2,:))],I_raw_all{3}(2,:))
plot([6001:6000+length(I_raw_all{4}(2,:))],I_raw_all{4}(2,:))
ylabel('Intensity (a.u.)')

yyaxis right
hold on
plot([1:length(I_raw_all{1}(2,:))],Depth_all{1}(2,:))
plot([2001:2000+length(I_raw_all{2}(2,:))],Depth_all{2}(2,:))
plot([4001:4000+length(I_raw_all{3}(2,:))],Depth_all{3}(2,:))
plot([6001:6000+length(I_raw_all{4}(2,:))],Depth_all{4}(2,:))
ylabel('Distance (m)')

title('Forehead')

figure()

yyaxis left
hold on
plot([1:length(I_raw_all{1}(4,:))],I_raw_all{1}(4,:))
plot([2001:2000+length(I_raw_all{2}(4,:))],I_raw_all{2}(4,:))
plot([4001:4000+length(I_raw_all{3}(4,:))],I_raw_all{3}(4,:))
plot([6001:6000+length(I_raw_all{4}(4,:))],I_raw_all{4}(4,:))
ylabel('Intensity (a.u.)')

yyaxis right
hold on
plot([1:length(I_raw_all{1}(4,:))],Depth_all{1}(4,:))
plot([2001:2000+length(I_raw_all{2}(4,:))],Depth_all{2}(4,:))
plot([4001:4000+length(I_raw_all{3}(4,:))],Depth_all{3}(4,:))
plot([6001:6000+length(I_raw_all{4}(4,:))],Depth_all{4}(4,:))
ylabel('Distance (m)')

title('Left Cheek')

figure()

yyaxis left
hold on
plot([1:length(I_raw_all{1}(5,:))],I_raw_all{1}(5,:))
plot([2001:2000+length(I_raw_all{2}(5,:))],I_raw_all{2}(5,:))
plot([4001:4000+length(I_raw_all{3}(5,:))],I_raw_all{3}(5,:))
plot([6001:6000+length(I_raw_all{4}(5,:))],I_raw_all{4}(5,:))
ylabel('Intensity (a.u.)')

yyaxis right
hold on
plot([1:length(I_raw_all{1}(5,:))],Depth_all{1}(5,:))
plot([2001:2000+length(I_raw_all{2}(5,:))],Depth_all{2}(5,:))
plot([4001:4000+length(I_raw_all{3}(5,:))],Depth_all{3}(5,:))
plot([6001:6000+length(I_raw_all{4}(5,:))],Depth_all{4}(5,:))
ylabel('Distance (m)')

title('Right Cheek')


%%
figure()

hold on
plot([1:length(I_raw_all{1}(2,:))],I_comp_norm_all{1}(1,:))
plot([2001:2000+length(I_raw_all{2}(2,:))],I_comp_norm_all{2}(1,:))
plot([4001:4000+length(I_raw_all{3}(2,:))],I_comp_norm_all{3}(1,:))
plot([6001:6000+length(I_raw_all{4}(2,:))],I_comp_norm_all{4}(1,:))
ylabel('Intensity (a.u.)')

ylabel('Distance (mm)')

title('Nose')

figure()

hold on
plot([1:length(I_raw_all{1}(2,:))],I_comp_norm_all{1}(2,:))
plot([2001:2000+length(I_raw_all{2}(2,:))],I_comp_norm_all{2}(2,:))
plot([4001:4000+length(I_raw_all{3}(2,:))],I_comp_norm_all{3}(2,:))
plot([6001:6000+length(I_raw_all{4}(2,:))],I_comp_norm_all{4}(2,:))
ylabel('Intensity (a.u.)')

ylabel('Distance (mm)')

title('Forehead')

figure()

hold on
plot([1:length(I_raw_all{1}(2,:))],I_comp_norm_all{1}(4,:))
plot([2001:2000+length(I_raw_all{2}(2,:))],I_comp_norm_all{2}(4,:))
plot([4001:4000+length(I_raw_all{3}(2,:))],I_comp_norm_all{3}(4,:))
plot([6001:6000+length(I_raw_all{4}(2,:))],I_comp_norm_all{4}(4,:))
ylabel('Intensity (a.u.)')

ylabel('Distance (mm)')

title('Left Cheek')

figure()

hold on
plot([1:length(I_raw_all{1}(2,:))],I_comp_norm_all{1}(5,:))
plot([2001:2000+length(I_raw_all{2}(2,:))],I_comp_norm_all{2}(5,:))
plot([4001:4000+length(I_raw_all{3}(2,:))],I_comp_norm_all{3}(5,:))
plot([6001:6000+length(I_raw_all{4}(2,:))],I_comp_norm_all{4}(5,:))
ylabel('Intensity (a.u.)')

ylabel('Distance (mm)')

title('Right Cheek')


%%
dis1 = Depth_concat(2,:);
dis2 = Depth_concat(1,:);
t = (0:length(int1)-1)*T;

figure(1)
yyaxis left
plot(t,smooth(intraw1))
yyaxis right
plot(t,smooth(int1))

figure(2)
plot(t,smooth(dis1))

figure(3)
yyaxis left
plot(t,smooth(intraw2))
yyaxis right
plot(t,smooth(int2))

figure(4)
plot(t,smooth(dis2))

figure(5)
yyaxis left
plot(t,smooth(int1,13));
ylabel('Intensity on Forehead (a.u.)')
yyaxis right
plot(t,smooth(int2,13));
ylabel('Intensity on Nose (a.u.)')
xlabel('Time (s)')

for j=1:8
    xline(event_start(j),'--r',event1)
    xline(event_end(j),'--r',event2)
end

legend('Forehead','Nose')


figure(6)
yyaxis left
plot(t,smooth(int2,13))
ylabel('Intensity (a.u.)')
yyaxis right
plot(t,smooth(dis2,13))
ylabel('Distance (m)')
xlabel('Time (s)')
for j=1:8
    xline(event_start(j),'--r',event1)
    xline(event_end(j),'--r',event2)
end
legend('Intensity','Distance')
title('I vs D on nose')

figure(7)
yyaxis left
plot(t,smooth(int1,13))
ylabel('Intensity (a.u.)')
yyaxis right
plot(t,smooth(dis1,13))
ylabel('Distance (m)')
xlabel('Time (s)')
for j=1:8
    xline(event_start(j),'--r',event1)
    xline(event_end(j),'--r',event2)
end
legend('Intensity','Distance')
title('I vs D on forehead')

figure(8)
plot(t,smooth(int1,13)-smooth(int2,13))
xlabel('Time (s)')
ylabel('A.U.')
% xline(startT,'--r',event1)
% xline(endT,'--r',event2)
legend('Blood flow difference (Nose-Forehead)')


figure(10)
plot(t,smooth(dis1,11)-smooth(dis2,11))
xlabel('Time (s)')
ylabel('A.U.')
for j=1:8
    xline(event_start(j),'--r',event1)
    xline(event_end(j),'--r',event2)
end
legend('distance difference (Nose-Forehead)')


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
while j+L-1 < size(RR_sig_concat,2)
    spectrum=fft(int3(j:j+L-1));
    P2 = abs(spectrum./L);
    onesided = P2(1:L/2+1);
    onesided(2:end-1) = 2*onesided(2:end-1);
    f = Fs*(0:(L/2))/L*60;
    f_Filtered_range=f<50|f>200;
    onesided(f_Filtered_range)=0;
% HR peak locate
    [pks,loc]=findpeaks(onesided) ;
    [maxval,maxindex]=max(pks);
    HR_current = f(loc(maxindex));
    HR = [HR HR_current];
    
    spectrum=fft(RR_sig_concat(2,j:j+L-1));
    P2 = abs(spectrum./L);
    onesided = P2(1:L/2+1);
    onesided(2:end-1) = 2*onesided(2:end-1);
    f = Fs*(0:(L/2))/L*60;
    f_Filtered_range=f<5|f>35;
    onesided(f_Filtered_range)=0;
% HR peak locate
    [pks,loc]=findpeaks(onesided) ;
    [maxval,maxindex]=max(pks);
    RR_current = f(loc(maxindex));
    
    RR = [RR RR_current];
    if counter == 174
        figure()
        plot(f,onesided)
        xlim([0 220])
    end
        
    j = j+step;
    counter = counter+1;
%     
end

t_HR = (L_t/2:step_t:((length(HR)-1)*step_t+L_t/2));

figure()
hold on
plot(t_HR,HR);
for j=1:8
    xline(event_start(j),'--r',event1)
    xline(event_end(j),'--r',event2)
end

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
% EEG_filename = 'mindMonitor_2022-01-11--11-44-18.csv';
% EEG_table = readtable(EEG_filename,'NumHeaderLines',0);
% EEG_properties = EEG_table.Properties;
% 
% timestamp = table2array(EEG_table(:,1));
% timestamp = datenum(timestamp);
% timestamp = (timestamp-timestamp(1))*24*3600;
% 
% time_remap = (0:size(timestamp)-1)/256;
% 
% figure(1)
% hold on;
% plot(time_remap,EEG_table.RAW_TP9+100);
% plot(time_remap,EEG_table.RAW_TP10);
% plot(time_remap,EEG_table.RAW_AF7-100);
% plot(time_remap,EEG_table.RAW_AF8-200);
% % 
% % TP: ear
% % AF: forehead
% time_remap_downspl = downsample(time_remap,10);
% tp9_Delta_downspl = downsample(EEG_table.Delta_TP9,10);
% tp10_Delta_downspl = downsample(EEG_table.Delta_TP10,10);
% af7_Delta_downspl = downsample(EEG_table.Delta_AF7,10);
% af8_Delta_downspl = downsample(EEG_table.Delta_AF8,10);
% 
% tp9_Beta_downspl = downsample(EEG_table.Beta_TP9,10);
% tp10_Beta_downspl = downsample(EEG_table.Beta_TP10,10);
% af7_Beta_downspl = downsample(EEG_table.Beta_AF7,10);
% af8_Beta_downspl = downsample(EEG_table.Beta_AF8,10);
% 
% tp9_Alpha_downspl = downsample(EEG_table.Alpha_TP9,10);
% tp10_Alpha_downspl = downsample(EEG_table.Alpha_TP10,10);
% af7_Alpha_downspl = downsample(EEG_table.Alpha_AF7,10);
% af8_Alpha_downspl = downsample(EEG_table.Alpha_AF8,10);
% 
% % figure()
% % plot(time_remap_downspl,smooth(tp10_thetaratio_downspl,100),'LineWidth',1.5);
% % plot(time_remap_downspl,smooth(tp9_thetaratio_downspl,100),'LineWidth',1.5);
% 
% 
% figure(2)
% hold on
% plot(time_remap_downspl,smooth(tp10_Delta_downspl+1,100));
% plot(time_remap_downspl,smooth(tp9_Delta_downspl+0.5,100));
% plot(time_remap_downspl,smooth(af7_Delta_downspl,100));
% plot(time_remap_downspl,smooth(af8_Delta_downspl-0.5,100));
% xline(event1_t,'--r',event1)
% xline(event2_t,'--r',event2)
% legend('tp10','tp9','af7','af8')
% title('Delta Wave')
% 
% figure(3)
% hold on
% plot(time_remap_downspl,smooth(tp10_Alpha_downspl,100)+1);
% plot(time_remap_downspl,smooth(tp9_Alpha_downspl,100)+0.5);
% plot(time_remap_downspl,smooth(af7_Alpha_downspl,100));
% plot(time_remap_downspl,smooth(af8_Alpha_downspl,100)-0.5);
% xline(event1_t,'--r',event1)
% xline(event2_t,'--r',event2)
% legend('tp10','tp9','af7','af8')
% title('Alpha Wave')
% 
% figure(4)
% hold on
% plot(time_remap_downspl,smooth(tp10_Beta_downspl,100));
% plot(time_remap_downspl,smooth(tp9_Beta_downspl,100));
% plot(time_remap_downspl,smooth(af7_Beta_downspl,100));
% plot(time_remap_downspl,smooth(af8_Beta_downspl,100));
% xline(event1_t,'--r',event1)
% xline(event2_t,'--r',event2)
% legend('tp10','tp9','af7','af8')
% title('Beta Wave')
% 
% beta_tp_avg = (smooth(tp10_Beta_downspl,100)+smooth(tp9_Beta_downspl,100))/2;
% delta_af_avg = (smooth(af7_Delta_downspl,100)+smooth(af8_Delta_downspl,100))/2;
% alpha_tp_avg = (smooth(tp10_Alpha_downspl,100)+smooth(tp10_Alpha_downspl,100))/2;
% 
% figure(5)
% hold on
% plot(time_remap_downspl,smooth(tp10_Beta_downspl,100));
% plot(time_remap_downspl,delta_af_avg);
% plot(time_remap_downspl,alpha_tp_avg);
% 
% xline(event1_t,'--r',event1)
% xline(event2_t,'--r',event2)
% legend('Beta wave (ear)','Delta wave (forehead)','Alpha wave (ear)');


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
dbv = smooth(int1,13)-smooth(int2,13);
filtered_dbv = lowpass(dbv,1e-10,Fs,'Steepness',0.999999);


figure(9)
yyaxis left
plot(t,smooth(dbv,50),'LineWidth',1.5)
xlabel('Time (s)')
ylabel('A.U.')

hold on
yyaxis right
plot(t_HR,HR);
ylabel('Heart Rate (bpm)')


for j=1:8
    xline(event_start(j),'--r',event1)
    xline(event_end(j),'--r',event2)
end


legend('Blood volume difference (Nose-Forehead)','Heart Rate');


legend('Blood volume difference (Nose-Forehead)');

figure(10)
plot(t,smooth(-intraw2+intraw1,50),'LineWidth',1.5)
xlabel('Time (s)')
ylabel('A.U.')

for j=1:8
    xline(event_start(j),'--r',event1)
    xline(event_end(j),'--r',event2)
end
legend('Blood volume difference (Nose-Forehead)')

figure(11)
plot(t,smooth(dbv,50),'LineWidth',1.5)
xlabel('Time (s)')
ylabel('A.U.')

for j=1:8
    xline(event_start(j),'--r',event1)
    xline(event_end(j),'--r',event2)
end
legend('Blood volume difference (Nose-Forehead)')

%% Calculate SNR
% 
% filtered_dbv = lowpass(dbv,1e-10,Fs,'Steepness',0.999999);
% 
% figure()
% hold on
% plot(t(300:end-600),filtered_dbv(300:end-600),'LineWidth',2)
% 
% yyaxis right
% hold on
% plot(HR_time,HR_ref,'--k')
% xlim([-100 1100]);
% 
% dbv_std = []
% dbv_len = length(dbv);
% for i = 300:(dbv_len-1500);
%     dbv_std(i-299) = std(filtered_dbv(i:i+900));
%     
% end
% 
