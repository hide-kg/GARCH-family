clc
clear
close all

load('TOPIX17_RV_01.mat')
load('EPU.mat')
load('sample_in_month.mat')

daily_return = daily_return_o2c;
test_start = cumsum(sample_in_month);

L = 1;
cumsample = cumsum(sample_in_month);
for i = 1:1708
    if i <= cumsample(L)
        e(i) = epu(L);
    else
        L = L + 1;
        e(i) = epu(L);
    end
end

epu_input = e';

period = 22;

% t1 = 735;
% t2 = 1490;

t1 = 1000;
t2 = length(daily_return);
[estimpara, forecast_fit, logL] = garch_midas_epu(daily_return(t1:t2), log(epu_input(t1:t2)), period, 0);


f = figure;
set(f, 'WindowStyle', 'Docked');
subplot(4,1,1)
plot(RV(t1:t2), 'b')
hold on
plot(forecast_fit.cond_vol, 'r', 'LineWidth', 1.5)
legend({'RV', '—\‘ª’l(GARCH-MIDAS)'})

subplot(4,1,2)
plot(RV(t1:t2), 'b')
hold on
plot(forecast_fit.short,'r', 'LineWidth', 1.5)
legend({'RV', '’ZŠú¬•ª'})

subplot(4,1,3)
plot(forecast_fit.short,'r')
hold on
plot(forecast_fit.long,'b', 'LineWidth',1.5)
legend({'’ZŠú¬•ª', '’·Šú¬•ª'})

subplot(4,1,4)
plot(forecast_fit.long,'b', 'LineWidth',1.5)



