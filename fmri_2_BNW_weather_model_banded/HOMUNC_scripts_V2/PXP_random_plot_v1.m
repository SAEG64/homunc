function [BMS] = PXP_random_plot_v1( bicE, title_tag ) % minus has to be added!

[BMS.alpha( 1, :), BMS.exp_r( 1, :), BMS.xp( 1, :), BMS.pxp( 1, :), BMS.bor( 1, :)] = spm_BMS( bicE );
ex1 = BMS.pxp( 1, :);
nn = size( ex1, 2);
ex1_rev = zeros(1,nn);
for ii = 1:nn; 
    ex1_rev(1,ii) = ex1(1,nn+1-ii);
end
figure
barh( ex1_rev, 'k' )
xlim([0,1])
set(gca,'XTick',[0:0.5:1])
title( title_tag )