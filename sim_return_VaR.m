load('return_2014010120201231.mat')
load('date_20142020.mat')
price_return = diff(log(return_2014010120201231(:,1)))*100;
distribution = 'norm';
[sigma, epsilon, mu, rho, omega, alpha, beta, nu, sim_r] = garch_estimate_vol2(price_return, distribution);



%% VaR‚ÌŒvŽZ

date_time = datetime(date, 'InputFormat', 'yyyy/MM/dd');

for i = 1:length(price_return)
    sim_r_sort(i,:) = sort(sim_r(i,:), 'descend');
end
for t = 1:length(price_return)
    VaR90(t,1) = quantile(sim_r_sort(t,:), 0.1);
    VaR99(t,1) = quantile(sim_r_sort(t,:), 0.01);
end
%%
t = 987;
close all
figure
plot(date_time(t:end), price_return(t-1:end),'r','LineWidth',1.5)
hold on
plot(date_time(t:end), sigma(t-1:end),'b--','LineWidth',1.5)
hold on
plot(date_time(t:end), VaR99(t-1:end),'k-.', 'LineWidth', 1.5)
grid on 
xlabel('ŠÏ‘ª“ú')
ylim([-40, 70])
ax = gca;
ax.FontSize = 20;


