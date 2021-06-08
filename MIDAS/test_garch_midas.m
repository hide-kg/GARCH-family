
clear


load('TOPIX17_RV_01.mat')
load('EPU.mat')
load('sample_in_month.mat')

daily_return = daily_return_o2c;
test_start = cumsum(sample_in_month);

period = 22;

[estimpara, forecast_fit, logL] = garch_midas(daily_return, 22, 1);



l = forecast_fit.long;
s = forecast_fit.short;

f = figure;
set(f, 'WindowStyle', 'Docked');
subplot(3,1,1)
plot(RV, 'b')
hold on
plot(forecast_fit.cond_vol, 'r', 'LineWidth', 1.5)
legend({'RV', '—\‘ª’l(GARCH-MIDAS)'})

subplot(3,1,2)
plot(RV, 'b')
hold on
plot(s, 'r', 'LineWidth', 1.5)
legend({'RV', '’ZŠú¬•ª'})

subplot(3,1,3)
plot(s,'r')
hold on
plot(l,'b', 'LineWidth',1.5)
legend({'’ZŠú¬•ª', '’·Šú¬•ª'})

%{
stat_time = 752;
qlike_garch_midas = mean(loss_function(RV(stat_time:end), forecast_fit.cond_vol(stat_time:end), 0))
stein_garch_midas = mean(loss_function(RV(stat_time:end), forecast_fit.cond_vol(stat_time:end), 1))
mse_garch_midas = mean(loss_function(RV(stat_time:end), forecast_fit.cond_vol(stat_time:end), 3))


%}
