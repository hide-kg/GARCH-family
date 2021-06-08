function [sigma, epsilon, mu, rho, omega, alpha, beta, nu, sim_r] = garch_estimate_vol2(price_return, distribution)

numData = length(price_return);
if distribution == 'norm'
    para0 = [0.01, 0.01, 0.01, 0.2, 0.7];
elseif distribution == 't'
    para0 = [0.01, 0.01, 0.01, 0.2, 0.7, 3];
end
llh = @(x0) garch_llh_vol2(x0, price_return,distribution);

[para,llh] = fminunc(llh, para0);

mu = para(1);
rho = para(2);
omega = para(3);
alpha = para(4);
beta = para(5);


if distribution == 't'
    nu = para(6)^2;
elseif distribution == 'norm'
    nu = [];
end



sigma = zeros(numData,1);
epsilon(1) = price_return(1)-mu;
for t = 2:numData
    epsilon(t) = price_return(t) - mu - rho * price_return(t-1);
end
epsilon = epsilon';

sigma(1) = mean(epsilon.^2);

for t = 2:numData
    sigma(t) = omega + alpha.*epsilon(t-1).^2 + beta.*sigma(t-1);
end
samplesize = 1000;
for t = 2:numData
    sim_epsilon(t,:) = sqrt(sigma(t)) * normrnd(0, 1, [1, samplesize]);
    sim_r(t,:) = mu + sim_epsilon(t,:);
end
