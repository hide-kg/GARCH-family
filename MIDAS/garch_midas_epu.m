function [estimpara, forecast_fit, logL, std_error, tstat, pval] = garch_midas_epu(daily_return, EPU, period, logtau)
%
% GARCH-MIDAS which is proposed by Engle et al. (2013)
% Refer to GARCHMIDAS.m
%
% Input :
%   daily_return : 日次収益率
%   EPU : EPU (daily_returnと同じサイズのベクトル)
%   period : 1ヶ月の観測日数
%   logtau : 長期成分で対数を取るかどうか

para0 = [0.01, 0.1, 0.8, 0.01, 0.1, 5];

% RVをあらかじめ計算しておく
% RVは1ヶ月間の日次収益率の二乗和
nobs = numel(daily_return);
nMonth = ceil(nobs/period);
Y = NaN(period, nMonth);
Y(1:nobs) = daily_return(:);
EPUmat = NaN(period, nMonth);
EPUmat(1:nobs) = EPU(:);
RV = nanmean(EPUmat,1);
if mod(nobs, period) > 0
    RV(end) = nanmean(EPUmat(end - period + 1:end));
end

nlag = 10;
llh = @(x0) -garch_llh(x0, Y, RV, nobs, nlag, logtau);

%para = fminunc(llh, para0);


% para0 = [mu, alpha, beta, m, theta, omega];
lb = [-Inf 0 0 -Inf -Inf 1.001];
ub = [Inf 1 1 Inf Inf 50];
para = fmincon(llh, para0, [], [], [], [], lb, ub, []);

mu = para(1);
alpha = para(2);
beta = para(3);
m = para(4);
theta = para(5);
%omega1 = para(6);
omega2 = para(6);

[llh, llhs, sigma, shortrun, longrun] = garch_llh(para, Y, RV, nobs, nlag, logtau);

fun = @(x0) garch_llh(x0, Y, RV, nobs, nlag, logtau);
VCV = vcv(fun, para);
tstats = para./sqrt(diag(VCV)');
se = sqrt(diag(VCV)');
pvals = 1/2 * erfc(0.7071 * abs(tstats)) * 2;
pvals(pvals < 1e-6) = 0;

n = length(para);
aic = -2 * llh + 2 * n;
bic = -2 * llh + n * log(nobs);

estimpara = struct();
estimpara.return = mu;
estimpara.garch = [alpha, beta];
estimpara.midas = [m, theta, omega2];

forecast_fit = struct();
forecast_fit.cond_vol = sigma;
forecast_fit.long = longrun;
forecast_fit.short = shortrun;

logL = struct();
logL.llh = llh;
logL.aic = aic;
logL.bic = bic;

std_error = struct();
std_error.return = se(1);
std_error.garch = se(2:3);
std_error.midas = se(4:end);

tstat = struct();
tstat.return = tstats(1);
tstat.garch = tstats(2:3);
tstat.midas = tstats(4:end);

pval = struct();
pval.return = pvals(1);
pval.garch = pvals(2:3);
pval.midas = pvals(4:end);



% 結果の出力
fprintf('Log likelihood : %12.6g\n', llh);
fprintf('AIC : %12.6g\n', aic);
fprintf('BIC : %12.6g\n', bic);
columnNames = {'Coeff', 'StdError', 'tStat', 'Pval'};
rowNames = {'mu', 'alpha', 'beta', 'm', 'theta', 'omega'};
Table = table(para', se', tstats', pvals', 'RowNames', rowNames, 'VariableNames', columnNames);
disp(Table)


end

function [llh, llhs, sigma, shortrun, longrun] = garch_llh(para0, Y, RV, nobs, nlag, logtau)
%
% input -
%   para0 : parameter vector
%   daily_return : freq-by-nMonth matrix form of observations (nobs = freq * nMonth)
%   RV : 1-by-nMonth realized volatility 
%   nobs : number of observations
%   olag : number of MIDAS lags

[period, nMonth] = size(Y);

mu = para0(1);
alpha = para0(2);
beta = para0(3);
m = para0(4);
theta = para0(5);
%omega1 = para0(6);
omega2 = para0(6);

if alpha<0 || beta<0 || alpha > 1 || beta > 1 || alpha + beta > 1
    % 上の尤度関数の計算の時点でマイナスをつけているので, ここではinfとする. 
    llhs = -inf(nobs,1);
    llh = sum(llhs);
    sigma = NaN(nobs, 1);
    shortrun = NaN(nobs, 1);
    longrun = NaN(nobs, 1);
    return
end

intercept = 1 - alpha - beta;

epsilon = Y - mu;
epsilon2 = epsilon .* epsilon;

shortrun = ones(period, nMonth);

% 初期値の計算
if logtau == 1
    tauAvg = exp(m + theta .* nanmean(RV));
else
    tauAvg = m + theta .* nanmean(RV);
end
sigma = tauAvg .* ones(period, nMonth);

weights = flip(beta_weight(nlag, omega2)');

for t = nlag+1:nMonth
    % 長期成分の計算
    % refer to Eq(5) in Engle et al. (2013)
    RVuse = RV(t - nlag:t-1);
    if logtau == 1
        tau = exp(m + theta .* (RVuse * weights));
    else
        tau = m + theta .* (RVuse * weights);
    end
    % Eq(4)の係数を計算している
    alphatau = alpha./tau;
    
    % 短期成分を計算
    % refer to Eq(4) in Engle et al. (2013)
    for n = 1:period
        ind = (t-1) * period + n;
        shortrun(ind) = intercept + alphatau .* epsilon2(ind-1) + beta .* shortrun(ind-1);
    end
    
    % 条件付き分散を計算
    % refer to Eq(3) in Engle et al. (2013)
    sigma(:,t) = tau .* shortrun(:,t);
end

% 対数尤度関数の計算
llh_mat = -1/2 .* (log(2*pi*sigma) + epsilon2./sigma);
llh_mat(:,1:nlag) = 0;
llhs = reshape(llh_mat(1:nobs), nobs, 1);
llh = sum(llhs);

% 条件付き分散を列ベクトルに変換
sigma = reshape(sigma(1:nobs), nobs, 1);
shortrun = reshape(shortrun(1:nobs), nobs, 1);
longrun = sigma ./ shortrun;





end

function phi_ell = beta_weight(nlag, gamma2)
% MIDAS項のbeta weight
% omega > 1である必要がある

gamma1 = 1;
j = 1:nlag;

phi_ell_upp = ((j./nlag).^(gamma1-1)) .* ((1 - j./nlag).^(gamma2-1));
phi_ell_low = sum(((j./nlag).^(gamma1-1)) .* (1 - j./nlag).^(gamma2-1));

phi_ell = phi_ell_upp./phi_ell_low;

end

function [VCV,scores,gross_scores]=vcv(fun,theta,varargin)
% Compute Variance Covariance matrix numerically only based on gradient
%
% USAGE:
%     [VCV,A,SCORES,GROSS_SCORES]=vcv(FUN,THETA,VARARGIN)
%
% INPUTS:
%     FUN           - Function name ('fun') or function handle (@fun) which will
%                       return the sum of the log-likelihood (scalar) as the 1st output and the individual
%                       log likelihoods (T by 1 vector) as the second output.
%     THETA         - Parameter estimates at the optimum, usually from fmin*
%     VARARGIN      - Other inputs to the log-likelihood function, such as data
%
% OUTPUTS:
%     VCV           - Estimated robust covariance matrix (see White 1994)
%     SCORES        - T x num_parameters matrix of scores
%     GROSS_SCORES  - Numerical scores (1 by num_parameters) of the objective function, usually for diagnostics
%
% COMMENTS:
%     For (Q)MLE estimation

% Michael Stollenwerk
% michael.stollenwerk@live.com
% 05.02.2019

% Adapted from robustvcv by Kevin Sheppard:
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 9/1/2005

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(theta,1)<size(theta,2)
    theta=theta';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


k=length(theta);
h=abs(theta*eps^(1/3));
h=diag(h);

% funの出力引数によって, ここの出力引数を変更する
% likeは1期間ごとの対数尤度となる
% 今, funはrealized_garch_llhで, 前から2つ目の出力引数はllhs(1期間ごとの対数尤度としている)
[~,like]=feval(fun,theta,varargin{:});

t=length(like);

LLFp=zeros(k,1);
LLFm=zeros(k,1);
likep=zeros(t,k);
likem=zeros(t,k);
for i=1:k
    thetaph=theta+h(:,i);
    [LLFp(i),likep(:,i)]=feval(fun,thetaph,varargin{:});
    thetamh=theta-h(:,i);
    [LLFm(i),likem(:,i)]=feval(fun,thetamh,varargin{:});
end

scores=zeros(t,k);
gross_scores=zeros(k,1);
h=diag(h);
for i=1:k
    scores(:,i)=(likep(:,i)-likem(:,i))./(2*h(i));
    gross_scores(i)=(LLFp(i)-LLFm(i))./(2*h(i));
end

B=cov(scores);
VCV=inv(B)/t;
end
