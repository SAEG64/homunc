%% HOMeostasis UNCertainty
% this script MODELs the data
% clean up 2024.05.04 by christoph.w.korn@gmail.com
clear
close all


%% directories & selection/deletion options
fs              = filesep;
which_sample    = 1; % 1=behavioral sample; 2=fMRI sample
dir_base        = pwd;
dir_scripts     = 'HOMUNC_scripts_v2';
cd([dir_base])


%% samples with transformed data
% B is the second step for the behavioral data: transformation
if which_sample == 1
    DATA = load('HOMUNC_data_beh_B_pilot_v1.mat');
elseif which_sample == 2
    DATA = load('HOMUNC_data_beh_B_fMRI_v1.mat');
end


%% some model-free plots
fosi1   = 16;
lw      = 3;
lw2     = 1.5;
% y_ticktock = [0:0.5:1];
% y_ticklimi = [0 1];

x_ticklimi = [0 1];
        
w_col = 22;
% w_col = 10; % 11
% y_ticktock = [600:200:1200];
% y_ticklimi = [600 1200];
y_ticktock = -0.4:0.2:1;
y_ticklimi = [-0.4 1];

% % bigger plots
% a = DATA.rule_sumi;
% mean_a = mean( a );
% sem_a  = std( a ) / sqrt( size( a, 1 ) );
% figure
% errorbar( mean_a, sem_a )

figure
x = 0.1:0.1:0.9;

a = squeeze(DATA.rule_all_56(2:10,w_col,:))';
mean_a = mean( a );
sem_a  = std( a ) / sqrt( size( a, 1 ) );

b = DATA.sumi_all_56(:,2:10);
mean_b = mean( b );
sem_b  = std( b ) / sqrt( size( b, 1 ) );


hold on
y = mean_a;
for i_p = 1:size(x,2)
    plot(x(1,i_p),y(1,i_p),'-o','MarkerSize',mean_b(1,i_p),'MarkerEdgeColor','black','MarkerFaceColor',[0.8 0.8 0.8],'LineWidth',lw);    
end
errorbar( x, mean_a, sem_a )


a = squeeze(DATA.rule_WWS_54(2:10,w_col,:))';
mean_a = nanmean( a );
sem_a  = nanstd( a ) / sqrt( size( a, 1 ) );

b = DATA.sumi_WWS_54(:,2:10);
mean_b = mean( b );
sem_b  = std( b ) / sqrt( size( b, 1 ) );


y = mean_a;
for i_p = 1:size(x,2)
    plot(x(1,i_p),y(1,i_p),'-o','MarkerSize',mean_b(1,i_p),'MarkerEdgeColor','black','MarkerFaceColor',[0.8 0.8 0.8],'LineWidth',lw);    
end
errorbar( x, mean_a, sem_a )


a = squeeze(DATA.rule_one_14(2:10,w_col,:))';
mean_a = nanmean( a );
sem_a  = nanstd( a ) / sqrt( size( a, 1 ) );

b = DATA.sumi_one_14(:,2:10);
mean_b = mean( b );
sem_b  = std( b ) / sqrt( size( b, 1 ) );
y = mean_a;
for i_p = 1:size(x,2)
    plot(x(1,i_p),y(1,i_p),'-o','MarkerSize',mean_b(1,i_p),'MarkerEdgeColor','black','MarkerFaceColor',[0.8 0.8 0.8],'LineWidth',lw);    
end
errorbar( x, mean_a, sem_a )

% axes etc.
title(['sample ' num2str(which_sample) ' 56'])
set(gca,'LineWidth', lw2, 'YLim', y_ticklimi, 'YTick', y_ticktock , 'XLim', x_ticklimi, 'FontName', 'Arial', 'FontSize', fosi1)

% plot(x(1,i_p),y(1,i_p),'-o','MarkerSize',m(1,i_p),'MarkerEdgeColor','black','MarkerFaceColor',[0.8 0.8 0.8],'LineWidth',lw);
% % Empirical: markers: mean data
%     for i_p = 1:size(x,2)
%         erplot = plot(x(1,i_p),y(1,i_p),'-o','MarkerSize',m(1,i_p),'MarkerEdgeColor','black','MarkerFaceColor',[0.8 0.8 0.8],'LineWidth',lw);
%         hold on
%     end
%     % error bars of data
%     plot([x; x], [y-err; y+err], '-k','LineWidth',2)


a = squeeze(DATA.rule_all_56(2:10,22,:))';
mean_a = mean( a );
sem_a  = std( a ) / sqrt( size( a, 1 ) );
figure
errorbar( mean_a, sem_a )
title(['sample ' num2str(which_sample) ' 56'])
xlim([0.5 10.5])
hold on 
a = squeeze(DATA.rule_WWS_54(2:10,22,:))';
mean_a = nanmean( a );
sem_a  = nanstd( a ) / sqrt( size( a, 1 ) );
errorbar( mean_a, sem_a )
a = squeeze(DATA.rule_one_14(2:10,22,:))';
mean_a = nanmean( a );
sem_a  = nanstd( a ) / sqrt( size( a, 1 ) );
errorbar( mean_a, sem_a )
plot( zeros(1,9) )



figure; scatter(DATA.data_R(:,22),DATA.data_R(:,14))
figure; scatter(DATA.data_R(:,22),DATA.data_R(:,54))
figure; scatter(DATA.data_R(:,22),DATA.data_R(:,56))

figure; scatter(DATA.data_R(:,22),DATA.data_R(:,10))
% figure; scatter(DATA.data_R(:,22),DATA.data_R(:,10).^2)


hist_daen_mean = mean( DATA.hist_daen, 3 );
figure; imagesc( hist_daen_mean )

[r,p] = corr( DATA.data_R(:,[22,14,54,56]));
figure; imagesc( r )
% blub

%% loop for analyses
if which_sample == 1
    
    mod_index_1 = [22,14,13,25,6,37,7,17,20,24,40,41,54,56]; % 55,
    mod_index_2 = [22,14,13,25,6,37,7,17,20,24,40,41,   56];
    
elseif which_sample == 2
    
    mod_index_1 = [22,14,13,25,6,37,7,17,20,24,40,41,54,56]; % 18 not included here
    mod_index_2 = [22,14,13,25,6,37,7,17,20,24,40,41,   56];
  
end

mod_index_1_plus    = [mod_index_1, 47,48,46,26,11];
mod_header_1      = DATA.header_columns( 1, mod_index_1 );
mod_header_1_plus = DATA.header_columns( 1, mod_index_1_plus );



delete_yn = 0;

for i_sub = 1:DATA.n_sub
    
    clear d
    d = DATA.data_back{ i_sub };
        
    d_answer(i_sub,1) = nanmean( d(:,11) );
    d_answer(i_sub,2) = d_answer(i_sub,1)>0.7  | d_answer(i_sub,1)<0.3;
    d_answer(i_sub,3) = d_answer(i_sub,1)>0.85 | d_answer(i_sub,1)<0.15;
    
    %% imortant here deletino of trials
    hori_del = 2;
    if delete_yn == 1        
        % only high or low wins
        % d( d(:,18) == 1, : ) = [];
        % higher number of days
        % d( d(:,17) ~= hori_del, : ) = [];
    end
    
    %% one modulator
    for i_mod = 1:size( mod_index_1, 2 )       
        
        x_regress_1                         = [ d(:,mod_index_1( i_mod )) ];
        y_regress_1                         = d(:,11) + 1;
        [reg_B_1 , reg_dev_1, reg_stats_1]  = mnrfit(x_regress_1, y_regress_1);
        SUB_B_1{i_mod}(i_sub, :)            = reg_B_1';
        SUB_p_1{i_mod}(i_sub, :)            = reg_stats_1.p';
        SUB_reg_dev_1(i_sub, i_mod)         = reg_dev_1;
        R0_1                                = corrcoef(x_regress_1);
        VIF_1{i_mod}(i_sub,:)               = diag(inv(R0_1))';
        
        AIC_1(i_sub, i_mod)                 = SUB_reg_dev_1(i_sub, i_mod) + 1*2; % Akaike Information Criterion
        BIC_1(i_sub, i_mod)                 = SUB_reg_dev_1(i_sub, i_mod) + 1*2*log( size(y_regress_1,1) ); % Bayesian Information Criterion: %% CORRECT BIC
        SIZE_check_1(i_sub, i_mod)          = size(y_regress_1,1);
        
    end % for i_mod
    
    %% two modulators
    for i_mod = 1:size( mod_index_2, 2 )        
        
        if which_sample == 1
            x_regress_2                         = [ d(:,54), d(:,mod_index_2( i_mod )) ]; % [ d(:,14), d(:,mod_index_2( i_mod )) ];
%             x_regress_2                         = [ d(:,14), d(:,mod_index_2( i_mod )) ]; % [ d(:,14), d(:,mod_index_2( i_mod )) ];
        elseif which_sample == 2
            x_regress_2                         = [ d(:,54), d(:,mod_index_2( i_mod )) ]; % [ d(:,22), d(:,mod_index_2( i_mod )) ];
        end
        
        y_regress_2                         = d(:,11) + 1;
        [reg_B_2 , reg_dev_2, reg_stats_2]  = mnrfit(x_regress_2, y_regress_2);
        SUB_B_2{i_mod}(i_sub, :)            = reg_B_2';
        SUB_p_2{i_mod}(i_sub, :)            = reg_stats_2.p';
        SUB_reg_dev_2(i_sub, i_mod)         = reg_dev_2;
        R0_2                                = corrcoef(x_regress_2);
        VIF_2{i_mod}(i_sub,:)               = diag(inv(R0_2))';
        
        AIC_2(i_sub, i_mod)                 = SUB_reg_dev_2(i_sub, i_mod) + 1*3; % Akaike Information Criterion
        BIC_2(i_sub, i_mod)                 = SUB_reg_dev_2(i_sub, i_mod) + 1*3*log( size(y_regress_2,1) ); % Bayesian Information Criterion: %% CORRECT BIC
        SIZE_check_2(i_sub, i_mod)          = size(y_regress_2,1);
        
    end % for i_mod
    
   
    
    %% RTs in subscript
%     homunc_RTs_v2 % to-be-changed
%     homunc_RTs_v3
    
    %% SPLIT
%     HOMUNC_split_v1
    
    %% do correlations
    [c_r(:,:,i_sub), c_p(:,:,i_sub)] = corr( d(:,mod_index_1_plus) );
    
end % for i_sub

%% mean parameters for all one- & two-variable models 
% these are needed for calculating uncertainties & discrepancies
% here they are calculated within the respective sample
% but they can also be taken from the behavioral sample for the fMRI sample
for i_mod = 1:size( SUB_B_1, 2 )
    
    SUB_B_1_mean(i_mod,:) = mean( SUB_B_1{i_mod} ); 
    
end
for i_mod = 1:size( SUB_B_2, 2 )
    
    SUB_B_2_mean(i_mod,:) = mean( SUB_B_2{i_mod} ); 
    
end











addpath('/home/sergej/spm12')

%% model comparison plots fomr HOMPRED
range_1 = 1:size(mod_index_1,2);
BIC_sum_1 = sum( BIC_1 );
BIC_fixed_plot_v1( BIC_sum_1(:,range_1), 'FIRST models' )
PXP_1 = PXP_random_plot_v1( -BIC_1(:,range_1)/2, 'FIRST models' );

range_2 = 1:size(mod_index_2,2);
BIC_sum_2 = sum( BIC_2 );
BIC_fixed_plot_v1( BIC_sum_2(:,range_2), 'SECOND models' )
PXP_2 = PXP_random_plot_v1( -BIC_2(:,range_2)/2, 'SECOND models' );


% range_3 = 1:size(mod_index_3,2);
% BIC_sum_3 = sum( BIC_3 );
% BIC_fixed_plot_v1( BIC_sum_3(:,range_3), 'THIRD models' )
% PXP_3 = PXP_random_plot_v1( -BIC_3(:,range_3)/2, 'THIRD models' );

BIC_comp_1_2 = [BIC_1(:,1), BIC_2(:,1)];
BIC_sum_comp_1_2 = sum( BIC_comp_1_2 );
BIC_fixed_plot_v1( BIC_sum_comp_1_2, 'FIRST vs SECOND models' )
PXP_1_2 = PXP_random_plot_v1( -BIC_comp_1_2/2, 'FIRST vs SECOND models' );

% BIC_comp_2_3 = [BIC_2(:,1), BIC_3(:,3)];
% BIC_sum_comp_2_3 = sum( BIC_comp_2_3 );
% BIC_fixed_plot_v1( BIC_sum_comp_2_3, 'SECOND vs THIRD models' )
% PXP_2_3 = PXP_random_plot_v1( -BIC_comp_2_3/2, 'SECOND vs THIRD models' );

% %% RT mean stuff
% i_RT = 1;
% SUB_RT_b_mean{ i_RT }(1,:) = mean( SUB_RT_b{ i_RT } );
% SUB_RT_b_mean{ i_RT }(2,:) = std( SUB_RT_b{ i_RT } ) / sqrt( size( SUB_RT_b{ i_RT }, 1) );
% for i_reg = 1:size(SUB_RT_b{ i_RT },2)
%     [thp(i_reg,1),thp(i_reg,2)] = ttest( SUB_RT_b{ i_RT }(:,i_reg) );
% end
% SUB_RT_B_sum = sum(SUB_RT_B);
% SUB_RT_B_sum_1 = SUB_RT_B_sum-SUB_RT_B_sum(1,1);







%% correlations mean
c_r_mean = mean( c_r, 3 );
figure
imagesc( c_r_mean )
c_r_mean = [c_r_mean; mod_index_1_plus];

%% SPLIT plot
% HOMUNC_split_plot_v1



% %% CAVE
% % P       = load([dir_base_1 fs dir_scripts fs 'PIL_mean_betas_v1.mat']); % this is for uncertainties, etc. --> to be checked later
% P = load('P_rapid_v24.mat'); % CAVE


%  %% CAVE
%         % uncertainties -- logic from hompred
%         % T.fit_pol{ i_sub }(:, 7:9 )  = sigmoid_derive( -T.fit_pol{ i_sub }(:, 1:3 ) );    
%         % difficulties according to mean for 56
%         d( i_line, 57 ) = P.SUB_B_1_mean( 15, 1 ) + ...
%             P.SUB_B_1_mean( 15, 2 ) * d( i_line, 56 );
%         d( i_line, 58 ) = sigmoid( -d( i_line, 57 ) ); 
%         d( i_line, 59 ) = sigmoid_derive( -d( i_line, 57 ) );
%         
%         % optimal policy
%         d( i_line, 60 ) = P.SUB_B_1_mean( 1, 1 ) + ...
%             P.SUB_B_1_mean( 1, 2 ) * d( i_line, 22 );
%         d( i_line, 61 ) = sigmoid( -d( i_line, 60 ) ); 
%         d( i_line, 62 ) = sigmoid_derive( -d( i_line, 60 ) );
        
delete_yn       = 0; % 0=no; 1=yes % deletion of specific trials for subanalyses as sepecified below
%% fMRI save
% cd('C:\Users\korn\Desktop\HH_HOMUNC_ana_2')
% cd('C:\HAMBURG\HH_ZURICH\HH_HOMUNC_fMRI\HH_HOMUNC_all\HH_HOMUNC_ana_2')
% save(['hori_' num2str(hori_del) '_samp_' num2str(which_sample)]);


% %% figure histogram of n days (all subjects)
% day_hist_mean = mean( days_hist );
% figure
% bar( day_hist_mean )
% title('N of trials for horizons 1:5')
% xlabel('days')
% ylabel('trials')



% %% INTERACTION model comparison
% % adapted from HOMPRED (ana_hom_use_v35_HH_all_two_models)
% for i_sub = 1:DATA.n_sub
%         
%     d = [];
%     d = DATA.data_back{ i_sub };
%     
%     y_regress   = d(:,11) + 1; % specify y
%     
%     % here slightly different logic: manually build larger x_regress and loop over it
% %     x_regress_I{1} = [ d(:,22), d(:,14) ];
% %     % optimal & p_forage
% %     x_regress_I{2} = [ d(:,22), d(:,14), d(:,22) .* d(:,14) ];
% %     % uncertainties
% %     % 'abs_conflict_30','diff_probs_31','diff_optimal_32'
% % %     Z.choice{i_ses}( t_trial, 8 ) = D.d( t_trial, 30); % conflict
% % %     Z.choice{i_ses}( t_trial, 9 ) = D.d( t_trial, 31); % difficulty_heuristic
% % %     Z.choice{i_ses}( t_trial,10 ) = D.d( t_trial, 32); % difficulty_optimal
% %     x_regress_I{3} = [ d(:,22), d(:,31) .* d(:,14)   ];
% %     x_regress_I{4} = [ d(:,32) .* d(:,22), d(:,14)   ];
% %     x_regress_I{5} = [ d(:,32) .* d(:,22), d(:,31) .* d(:,14)   ];
% %     x_regress_I{6} = [ d(:,22), (1-d(:,31)) .* d(:,14)   ];
% %     x_regress_I{7} = [ d(:,22), d(:,14), d(:,31)   ];
% %     x_regress_I{8} = [ d(:,22), (1-d(:,31)) .* d(:,14), d(:,14)  ];
% % %     x_regress_I{8} = [ d(:,22), d(:,14), d(:,18)   ];
% % %     x_regress_I{9} = [ d(:,22) .* d(:,18), d(:,14) ];
% % %     x_regress_I{3} = [ T_2{ i_sub }(whi_main,16),    T_2{ i_sub }(whi_main, 5) , T_2{ i_sub }(whi_main,16) .*    T_2{ i_sub }(whi_main, 5)  ];
% % %     x_regress_I{4} = [ T_2{ i_sub }(whi_main,16), (1-T_2{ i_sub }(whi_main,41)), T_2{ i_sub }(whi_main,16) .* (1-T_2{ i_sub }(whi_main,41)) ];
% % %     x_regress_I{5} = [ T_2{ i_sub }(whi_main,16),    T_2{ i_sub }(whi_main, 4) , T_2{ i_sub }(whi_main,16) .*    T_2{ i_sub }(whi_main, 4)  ];
% % %     % p_forage % others
% % %     x_regress_I{6} = [ T_2{ i_sub }(whi_main,14),    T_2{ i_sub }(whi_main,19) , T_2{ i_sub }(whi_main,14) .*    T_2{ i_sub }(whi_main,19)  ];
% % %     x_regress_I{7} = [ T_2{ i_sub }(whi_main,14),    T_2{ i_sub }(whi_main, 5) , T_2{ i_sub }(whi_main,14) .*    T_2{ i_sub }(whi_main, 5)  ];
% % %     x_regress_I{8} = [ T_2{ i_sub }(whi_main,14), (1-T_2{ i_sub }(whi_main,41)), T_2{ i_sub }(whi_main,14) .* (1-T_2{ i_sub }(whi_main,41)) ];
% % %     x_regress_I{9} = [ T_2{ i_sub }(whi_main,14),    T_2{ i_sub }(whi_main, 4) , T_2{ i_sub }(whi_main,14) .*    T_2{ i_sub }(whi_main, 4)  ];
%     
% %     x_regress_I{1} = [                d(:,22),                d(:,41)   ];
% %     x_regress_I{2} = [                d(:,22), (1-d(:,31)) .* d(:,14)   ];
% %     x_regress_I{3} = [ (1-d(:,32)) .* d(:,22),                d(:,14)   ];
% %     x_regress_I{4} = [ (1-d(:,32)) .* d(:,22), (1-d(:,31)) .* d(:,14)   ];
% %     
% %     x_regress_I{5} = [    d(:,31)  .* d(:,22), (1-d(:,31)) .* d(:,14)   ];
% %     x_regress_I{6} = [ (1-d(:,32)) .* d(:,22),    d(:,32)  .* d(:,14)   ];   
% %     x_regress_I{2} = [    d(:,31)  .* d(:,22), (1-d(:,31)) .* d(:,41)   ];
% %     x_regress_I{3} = [ (1-d(:,32)) .* d(:,22),    d(:,32)  .* d(:,41)   ];
% %     
% %     x_regress_I{4} = [ d(:,22), d(:,41), d(:,22).* (d(:,18)-2) ];
% %     x_regress_I{5} = [ d(:,22), d(:,41), d(:,22).* (d(:,18)-1) ];
% %     x_regress_I{6} = [ d(:,22), d(:,41), d(:,41).* (d(:,18)-2) ];
% %     x_regress_I{7} = [ d(:,22), d(:,41), d(:,41).* (d(:,18)-1) ];
% 
%     x_regress_I{1} = [ d(:,22), d(:,41) ];
%     x_regress_I{2} = [ d(:,41), d(:,22).* d(:,17) ];
%     x_regress_I{3} = [ d(:,41), d(:,22), d(:,22).* d(:,17) ];
%     x_regress_I{4} = [ d(:,22), d(:,41).* d(:,17) ];
%     x_regress_I{5} = [ d(:,22), d(:,41), d(:,41).* d(:,17) ];
%     x_regress_I{6} = [ d(:,22), d(:,41), d(:,17) ];
%     x_regress_I{7} = [ d(:,41), d(:,22).* d(:,48) ];
%     x_regress_I{8} = [ d(:,41), d(:,22), d(:,22).* d(:,48) ];
%     x_regress_I{9} = [ d(:,41), d(:,22), d(:,48) ];
%     
%     x_regress_I{10} = [ d(:,22), d(:,41).* d(:,47) ];
%     x_regress_I{11} = [ d(:,22), d(:,41), d(:,41).* d(:,47) ];
%     x_regress_I{12} = [ d(:,22), d(:,41), d(:,47) ];
%     x_regress_I{13} = [ d(:,22), d(:,41), d(:,53) ];
% %     if which_sample == 2
% %         x_regress_I{2} = [ d(:,41), d(:,22).* (d(:,18)-1), d(:,22).* (1-(d(:,18)-1)) ];
% %         x_regress_I{3} = [ d(:,22), d(:,41).* (d(:,18)-1), d(:,41).* (1-(d(:,18)-1)) ];
% %     end
%     % col 18: win-high-low does not exist in pilot
%     if which_sample == 1
%         length_interaction = length( x_regress_I );
%     elseif which_sample == 2
%         length_interaction = length( x_regress_I );
%     end
%     for i_mod = 1:length_interaction;
%         
% %         [SUB_B_I{i_mod}(i_sub, :), SUB_reg_dev_I(i_sub, i_mod), SUB_p_I{i_mod}(i_sub, :)] = mnrfit(x_regress_I{i_mod}, y_regress);        
%         BIC_I(i_sub, i_mod) = SUB_reg_dev_I(i_sub, i_mod) + (size(x_regress_I{i_mod},2)+1)*log( size(y_regress,1) ); % Bayesian Information Criterion
%         
%     end
% end
% 
% %% BIC_I model comparisons
% BIC_I_with_win = [ BIC_I  ]; % BIC_2(:,1), 
% BIC_I_with_win_sum = sum( BIC_I_with_win, 1 );
% BIC_I_with_win_sum = BIC_I_with_win_sum - BIC_I_with_win_sum(1,1);

% BIC_fixed_plot_v1( BIC_I_with_win_sum, 'INTERACTION models' )
% PXP_Inter = PXP_random_plot_v1( -BIC_I_with_win/2, 'INTERACTION models' );

% figure
% scatter( d(:,14), 1-d(:,31) )
% scatter( d(:,14), d(:,30) )
% scatter( d(:,22), d(:,32) )
% scatter( d(:,22), d(:,30) )
% scatter( d(:,31), d(:,30) )
% scatter( d(:,32), d(:,30) )

% figure
% scatter( d(:,22), d(:,48) )
% hold on
% scatter( d(:,22), d(:,45) )
% 
% figure
% scatter( d(:,41), d(:,47) )
% hold on
% scatter( d(:,41), d(:,43) )
% 
% figure
% scatter( d(:,14), d(:,52) )
% hold on
% scatter( d(:,14), d(:,50) )
% 
% figure
% scatter( d(:,17), d(:,22) ) % days & optimal
% hold on
% scatter( d(:,17), d(:,48) ) % days & difficulty optimal

% RT
% clear BMS
% i_sit = 1;
% [BMS{ 1,2 }.alpha( i_sit, :), BMS{ 1,2 }.exp_r( i_sit, :), BMS{ 1,2 }.xp( i_sit, :), BMS{ 1,2 }.pxp( i_sit, :), BMS{ 1,2 }.bor( i_sit, :)] = spm_BMS( SUB_RT_B ); 
% figure
% bar( BMS{ 1,2 }.pxp( i_sit, :) )

%% Save for uncertainties & conflict from pilot data only
% PAL.prob_B_mean = mean( SUB_B_1{2} ); % 14
% PAL.poli_B_mean = mean( SUB_B_1{1} ); % 22
% PAL.EVch_B_mean = mean( SUB_B_1{7} ); % 41 % mean( SUB_B_1{8} ); % 41
% 
% POL.prob_B_mean = [P.PIL.prob_B_mean; PAL.prob_B_mean];
% POL.poli_B_mean = [P.PIL.poli_B_mean; PAL.poli_B_mean];
% POL.EVch_B_mean = [P.PIL.EVch_B_mean; PAL.EVch_B_mean];
%% Save for posterior predictive checks
% PBE = SUB_B_2{ 2 };
% PRT = SUB_RT_b{ 1 };

%% RT
% i_RT = 1;
% SUB_RT_b_mean(i_RT,:) = nanmean( SUB_RT_b{i_RT} );
% SUB_RT_b_semi(i_RT,:) = nanstd ( SUB_RT_b{i_RT} ) / sqrt( size( SUB_RT_b{i_RT}, 1) );

% figure
% errorbar( SUB_RT_b_mean, SUB_RT_b_semi )
% title( 'SUB_B_full' )
% for i_tt_RT = 1:size(SUB_RT_b{i_RT},2)
%     [t_ful_RT(i_tt_RT,1), t_ful_RT(i_tt_RT,2)] = ttest( SUB_RT_b{i_RT}(:,i_tt_RT) );
% end

%% fMRI save
% cd('C:\Users\korn\Desktop\HH_HOMUNC_ana_2')
% cd('C:\HAMBURG\HH_ZURICH\HH_HOMUNC_fMRI\HH_HOMUNC_all\HH_HOMUNC_ana_2')
% save(['hori_' num2str(hori_del) '_samp_' num2str(which_sample)]);
% save('beh2fmri_homunc_v5A', 'data_pers_fmri', 'list_cat', 'R');

% %% New in v35: All models with two predictors
% mod_header_4 = mod_header_1( range_1 );
% mod_indece_4 = mod_index_1( range_1 ); % mod_indece_1( range_1 );
% for i_sub = 1:n_sub % 1:n_sub
%     i_get = 1;
% %     w_main    = 1:T.SUB_size( i_sub, 4 ); % select gambles
% %     y_regress = USE_2{ i_sub }(w_main,10) + 1; % specify y
%     d = [];
%     d = data_wivari{ i_sub };    
%     y_regress   = d(:,11) + 1; % specify y
%     
%     for i_mod_out = 1:length( mod_indece_4 )
%         for i_mod_inn = i_mod_out+1:length( mod_indece_4 )
%             
% %             x_regress            = [ USE_2{ i_sub }( w_main, mod_indece_4(i_mod_out) ), USE_2{ i_sub }( w_main, mod_indece_4(i_mod_inn) ) ];
% %             x_regress            = [ USE_2{ i_sub }( w_main, mod_indece_4(i_mod_out) ), USE_2{ i_sub }( w_main, mod_indece_4(i_mod_inn) ) ];
%             x_regress            = [ d(:,mod_indece_4(i_mod_out)), d(:,mod_indece_4(i_mod_inn)) ];
%             [SUB_B_4{i_get}(i_sub, :), SUB_reg_dev_4(i_sub, i_get), SUB_p_4{i_get}(i_sub, :)] = mnrfit(x_regress, y_regress);
%             BIC_4(i_sub, i_get) = SUB_reg_dev_4(i_sub, i_get) + (size(x_regress,2)+1)*log( size(y_regress,1) ); % Bayesian Information Criterion
%             % also arrange in table for easier copying
%             BIC_4_table( i_mod_out, i_mod_inn, i_sub ) = BIC_4(i_sub, i_get);
%             i_get = i_get + 1;
%             
%         end
%     end
% end
% BIC_4_sum = sum( BIC_4 );
% BIC_4_sum = BIC_4_sum - BIC_4_sum(1,1);
% 
% BIC_4_table_sum = sum( BIC_4_table, 3 );
% BIC_4_table_sum_diff = BIC_4_table_sum - BIC_4_table_sum(1,2); 
% BIC_4_table_sum_diff_round = round( BIC_4_table_sum_diff );
% 
% BIC_fixed_plot_v1( BIC_4_sum, 'FULL model space' )
% PXP_BIC_4 = PXP_random_plot_v1( -BIC_4, 'FULL model space' );
% PXP_BIC_4_H = PXP_random_plot_v1( -BIC_4/2, 'FULL model space' );
% 
% % put PXP into a table for easier copying
% i_get = 1;
% for i_mod_out = 1:length( mod_indece_4 )
%     for i_mod_inn = i_mod_out+1:length( mod_indece_4 )
%         PXP_BIC_4_H_table( i_mod_out, i_mod_inn ) = PXP_BIC_4_H.pxp(1,i_get);
%         i_get = i_get + 1;
%     end
% end
% PXP_BIC_4_H_table_round = round( PXP_BIC_4_H_table, 3 );

%% LD 


d_use_in = [];
per_forage = [];
per_unqVal = [];

depMod = 18; % Model I want to check

figure
for i_s = 1:DATA.n_sub
    
     d_sub = DATA.data_back{ i_s };
    
    % unique Values in modulator
    unqValues = unique(d_sub(:,depMod)); 
    
    
    for i_idx = 1:size(unqValues,1) % for each possible value in modulator

        % get percentage of foraging choices for particular modulator value
        i_val = unqValues(i_idx,1);
        d_use_in = d_sub(d_sub(:,depMod)== i_val,:);                        % only trials where value == i_val
        per_forage(i_s,i_idx) = sum(d_use_in(:,11))/size(d_use_in,1);       % how often foraged in trials with value == i_val
        
        % get percentage of trials with particular modulator value
        n_unqVal = sum(d_sub(:,depMod)== i_val);                            % absolute N of unique value of modulator
        n_allVal = size(d_sub(:,depMod));                                   % N trials (all values of modulator)
        per_unqVal(i_s,i_idx) = n_unqVal/n_allVal(1,1);                     % percentage of trials with unique value
        
    end
    
    % bar plot (percentage of foraging choices - modulator categories/values)
    subplot(6,5,i_s)
    %         h = bar([per_forage(i_s,:); rel_unqVal(i_s,:)]');
    h = bar(per_forage(i_s,:));
    h(1).FaceColor = [0 0.6 1];
    %         h(2).FaceColor = [0.4 0.8 1];
    title(['Sub ',num2str(i_s)]);
    xticklabels(unqValues)
    ylim([0,1])
    ylabel('Foraging choice')
%    sgtitle(DATA.header_columns{1,depMod},'Interpreter', 'none')
    %         legend('N choice','N trials')
end


myColormap = [
%     103,0,31;
    165,15,21;
    203,24,29;
    239,59,44;
%     251,106,74;
%     252,146,114;
    252,187,161;
    254,224,210;
%     255,245,240;
%     247,247,247; % white
    247,251,255;
    222,235,247;
    198,219,239;
    158,202,225;
    107,174,214;
    66,146,198;
    33,113,181;
    8,81,156;
    8,48,107];

myColormap = myColormap / 256; 


%%

fh3 = figure;
c_r_mean_l = (c_r_mean(range_1,range_1 )); % set lower triangular to zero
imagesc(c_r_mean_l)
title('Correlations')
set(gca, 'YTick', 1:14);
set(gca, 'XTick', 1:14); 
% yticklabels(newLabels(:,range_1)) % add new labels 
set(gca,'xticklabel',{[]})
set(gca,'yticklabel',{[]})
fh3.Position = [189,127,1051,851]; % size
set(gca, 'FontName', 'Calibri', 'FontSize', 32)
colormap(myColormap)
colorbar



fh4 = figure;
imagesc(eye(14))
colormap([1,1,1])
set(gca, 'YTick', 1:14);
set(gca, 'XTick', []); 
set(gca,'xticklabel',{[]})
set(gca,'yticklabel',{[]})
fh4.Position = [189,127,1051,851]; % size

%% INTERACTION model comparison

%% BIC_I model comparisons

%% LD 

% d_use_in = [];
% per_forage = [];
% per_unqVal = [];
% 
% depMod = 21; % Model I want to check
% 
% figure
% for i_s = 1:length( sub_use )
%     
%      d_sub = data_wivari{ i_s }; % data per sub
%     
%     % unique Values in modulator
%     unqValues = unique(d_sub(:,depMod)); 
%     
%     for i_idx = 1:size(unqValues,1) % for each possible value in modulator
% 
%         % get percentage of foraging choices for particular modulator value
%         i_val = unqValues(i_idx,1);
%         d_use_in = d_sub(d_sub(:,depMod)== i_val,:);                        % only trials where value == i_val
%         per_forage(i_s,i_idx) = sum(d_use_in(:,11))/size(d_use_in,1);       % how often foraged in trials with value == i_val
%         
%         % get percentage of trials with particular modulator value
%         n_unqVal = sum(d_sub(:,depMod)== i_val);                            % absolute N of unique value of modulator
%         n_allVal = size(d_sub(:,depMod));                                   % N trials (all values of modulator)
%         per_unqVal(i_s,i_idx) = n_unqVal/n_allVal(1,1);                     % percentage of trials with unique value
%         
%     end
%     
%     % bar plot (percentage of foraging choices - modulator categories/values)
%     subplot(6,5,i_s)
%     %         h = bar([per_forage(i_s,:); rel_unqVal(i_s,:)]');
%     h = bar(per_forage(i_s,:));
%     h(1).FaceColor = [0 0.6 1];
%     %         h(2).FaceColor = [0.4 0.8 1];
%     title(['Sub ',num2str(i_s)]);
%     xticklabels(unqValues)
%     ylim([0,1])
%     ylabel('Foraging choice')
%     sgtitle(header_columns{1,depMod},'Interpreter', 'none')
%     %         legend('N choice','N trials')
% end
