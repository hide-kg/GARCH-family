clear


load('TOPIX17_RV_02.mat')
load('EPU.mat')
load('sample_in_month.mat')

cumsample = cumsum(sample_in_month);

L = 1;

for i = 1:1708
    if i <= cumsample(L)
        e(i) = epu(L);
    else
        L = L + 1;
        e(i) = epu(L);
    end
end



y = daily_return_o2c;

[estParams,EstParamCov,Variance,LongRunVar,ShortRunVar,logL] = ...
    GarchMidas(y, 'X', e', 'Period', 22, 'LogTau', true);


figure
subplot(3,1,1)
plot(RV, 'b')
hold on
plot(Variance, 'r', 'LineWidth', 1.5)
legend({'RV', '—\‘ª’l(GARCH-MIDAS)'})

subplot(3,1,2)
plot(RV, 'b')
hold on
plot(ShortRunVar, 'r', 'LineWidth', 1.5)
legend({'RV', '’ZŠú¬•ª'})

subplot(3,1,3)
yyaxis left
plot(ShortRunVar,'r')
hold on
plot(LongRunVar,'b-', 'LineWidth',1.5)
legend({'’ZŠú¬•ª','’·Šú¬•ª'})



%{
[estParams,EstParamCov,Variance,LongRunVar,ShortRunVar, logL] = ...
    GarchMidas(y, 'Period', 22);


figure
subplot(3,1,1)
plot(RV, 'b')
hold on
plot(Variance, 'r', 'LineWidth', 1.5)
legend({'RV', '—\‘ª’l(GARCH-MIDAS)'})

subplot(3,1,2)
plot(RV, 'b')
hold on
plot(ShortRunVar, 'r', 'LineWidth', 1.5)
legend({'RV', '’ZŠú¬•ª'})

subplot(3,1,3)
yyaxis left
plot(ShortRunVar,'r')
hold on
plot(LongRunVar,'b-', 'LineWidth',1.5)

legend({'’ZŠú¬•ª','’·Šú¬•ª'})
%}
