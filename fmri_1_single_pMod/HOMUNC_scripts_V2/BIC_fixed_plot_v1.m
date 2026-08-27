function BIC_fixed_plot_v1( bicF, title_tag )

nn = size( bicF, 2);
bicF_rev = zeros(1,nn);
for ii = 1:nn; 
    bicF_rev(1,ii) = bicF(1,nn+1-ii);
end
bicF_rev = bicF_rev - bicF_rev(nn);
figure
barh( bicF_rev, 'k' )
title( title_tag )

