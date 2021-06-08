function [estimpara, forecast_fit, logL] = garch(daily_return, test_start)

para0 = [0.01, 0.01, 0.01, 0.2, 0.7];

ver = 0;
llh = @(x0) -garch_llh(x0, daily_return(1:test_start-1), test_start, ver);

para = fminunc(llh, para0);

mu = para(1);
rho = para(2);
omega = para(3);
alpha = para(4);
beta = para(5);

ver = 1;
[llh, llhs, sigma] = garch_llh(para, daily_return, test_start, ver);

n = length(para);
aic = -2 * llh + 2 * n;
bic = -2 * llh + n * log(test_start-1);

estimpara = struct();
estimpara.return = [mu, rho];
estimpara.garch = [omega, alpha, beta];

forecast_fit = struct();
forecast_fit.cond_vol = sigma;

logL = struct();
logL.llh = llh;
logL.aic = aic;
logL.bic = bic;


end

function [llh, llhs, sigma, epsilon] = garch_llh(para0, daily_return, test_start, ver)

mu = para0(1);
rho = para0(2);
omega = para0(3);
alpha = para0(4);
beta = para0(5);

T = length(daily_return);

sigma = zeros(T, 1);
epsilon = zeros(T, 1);
llhs = zeros(T, 1);

epsilon(1) = daily_return(1) - mu;

for t = 2:T
    epsilon(t) = daily_return(t) - mu - rho * daily_return(t-1);
end

sigma(1) = mean(epsilon.^2);

for t = 2:T
    sigma(t) = omega + alpha * epsilon(t-1)^2 + beta * sigma(t-1);
    llhs(t) = -1/2 * (log(sigma(t)) + epsilon(t)^2/sigma(t));
end

if ver == 0
    llh = sum(llhs);
elseif ver == 1
    llh = sum(llhs(1:test_start-1));
end

if alpha<0 || beta<0 || alpha > 1 || beta > 1 || omega < 0 || alpha + beta > 1
    % 上の尤度関数の計算の時点でマイナスをつけているので, ここではinfとする. 
    llh = -inf;
end


end
