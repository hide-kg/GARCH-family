function [estimpara, forecast_fit, logL] = egarch(daily_return, test_start)
%
% 2021/4/14
%   EGARCHƒ‚ƒfƒ‹
%


para0 = [0.1, 0.8, 0.1, 0.1];

ver = 0;
llh = @(x0) -egarch_llh(x0, daily_return(1:test_start-1), test_start, ver);
[para] = fminunc(llh, para0);

omega = para(1);
beta = para(2);
theta = para(3);
gamma = para(4);

ver = 1;
[llh, h] = egarch_llh(para, daily_return, test_start, ver);

n = length(para);
aic = -2 * llh + 2 * n;
bic = -2 * llh + n * log(test_start-1);

estimpara = struct();
estimpara.egarch = [omega, beta, theta, gamma];

forecast_fit = struct();
forecast_fit.cond_vol = h;

logL = struct();
logL.llh = llh;
logL.aic = aic;
logL.bic = bic;

end


function [llh, h] = egarch_llh(para, daily_return, test_start, ver)

if ver == 1
    T = length(daily_return);
elseif ver == 0
    T = test_start-1;
end

omega = para(1);
beta = para(2);
theta = para(3);
gamma = para(4);

logh = zeros(T, 1);
z = zeros(T, 1);
llhs = zeros(T, 1);

logh(1) = log(mean(daily_return.^2));

for t = 2:T
    if t == 2
        z(1) = daily_return(1)./sqrt(exp(logh(1)));
    end
    if z(t-1) > 0
        logh(t) = omega + beta * logh(t-1) + (gamma + theta) * abs(z(t-1)) - gamma * sqrt(2/pi);
    elseif z(t-1) < 0
        logh(t) = omega + beta * logh(t-1) + (gamma - theta) * abs(z(t-1)) - gamma * sqrt(2/pi);
    end
    z(t) = daily_return(t)./sqrt(exp(logh(t)));
    llhs(t) = -1/2 * (logh(t) + (daily_return(t)^2)./exp(logh(t)));
end

h = exp(logh);

if ver == 0
    llh = sum(llhs);
elseif ver == 1
    llh = sum(llhs(1:test_start-1));
end

end

