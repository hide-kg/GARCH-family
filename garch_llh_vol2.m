function llh = garch_llh_vol2(para0, price_return, distribution)
%para0 - initial value of parameter
%       para0 = [mu0, omega0, alpha0, beta0]
%data - data of the return
mu = para0(1);
rho = para0(2);
omega = para0(3);
alpha = para0(4);
beta = para0(5);
if distribution == 't'
    nu = para0(6)^2;
end

numData = max(size(price_return));
sigma = zeros(numData,1);
epsilon = zeros(numData,1);
epsilon(1) = price_return(1) - mu;
for t = 2:numData
    epsilon(t) = price_return(t) - mu - rho * price_return(t-1);
end

sigma(1) = mean(epsilon.^2);
llh = 0;

if distribution == 'norm'
    for t = 2:numData
        sigma(t) = omega + alpha .* epsilon(t-1).^2 + beta .* sigma(t-1);
        llh = llh + 1/2 * log(sigma(t))+ 1/2 * epsilon(t).^2/sigma(t);
    end
elseif distribution == 't'
    for t = 2:numData
        sigma(t) = omega + alpha .* epsilon(t-1).^2 + beta .* sigma(t-1);
        llh = llh - log(gamma((nu+1)/2)) + log(gamma(nu/2)) + 0.5 * log(pi*(nu-2))...
            + 0.5 * log(sigma(t)) + ((nu+1)/2) * log(1 + (epsilon(t).^2/(sigma(t) * (nu-2))));
    end
end



if alpha<0 || beta<0 || alpha > 1 || beta > 1 || omega < 0 || alpha + beta > 1
    % 上の尤度関数の計算の時点でマイナスをつけているので, ここではinfとする. 
    llh = inf;
end

end
