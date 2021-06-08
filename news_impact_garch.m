price_return = daily_return;
distribution = 'norm';

[sigma, epsilon, mu, rho, omega, alpha, beta] = garch_estimate_vol2(price_return, distribution);

sample_var = var(epsilon);

for t = 2:length(daily_return)
    sigma_t(t) = omega + alpha * epsilon(t-1)^2 + beta * sample_var;
end
hold on
plot(epsilon(1:length(daily_return)-1), sigma_t(2:length(daily_return)), '.')
