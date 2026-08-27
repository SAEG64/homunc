%% HOMeostasis UNCertainty
% this script TRANSforms the data
% clean up 2024.05.04 by christoph.w.korn@gmail.com
clear
close all


%% directories & selection/deletion options
fs              = filesep;
which_sample    = 2; % 1=behavioral sample; 2=fMRI sample
dir_base        = pwd;
dir_scripts     = 'HOMUNC_scripts_v2';
cd([dir_base])

H = load('HOMUNC_data_beh_B_fMRI_v1', 'header_columns')
headers = H.header_columns'

%% samples with preprocessed data
% A is the first step for the behavioral data: preprocessing 
if which_sample == 1
    load('HOMUNC_data_beh_A_pilot_v1.mat')
elseif which_sample == 2
    load('HOMUNC_data_beh_A_fMRI_v1.mat')
end


%% LOAD lists for optimal policies, etc.
% get L_big = full item list (2025x8 double) 
L_load  = load([dir_base fs 'L_for_HOMUNC_list_v1.mat']);
L       = L_load.L_big;                                         % this is an "old" list going to be used below
L_REV   = load([dir_base fs 'L_REV_v2.mat']);    % this is a  "new" list going to be used below


%% loop for getting relevant variables into data
% get all columns for models, i.e., variants of optimal policies & heuristics

% preallocations
n_sub     = length(sub_use);
data_R    = []; % this is a concatenation of all data for R or the decision tree calculation
size_tria = NaN( n_sub,  5 ); % sizes of the trial numbers

data_back = cell( n_sub, 1 ); 
hist_days = NaN( n_sub,  5 );
hist_ener = NaN( n_sub,  5 );
hist_daen = zeros( 5, 6, n_sub );
% d_mean    = NaN( n_sub, 58 );
rule_vect = 0:0.1:1;
rule_n    = length( rule_vect );
rule_sumi = NaN( n_sub, rule_n );
d_count = [];

for i_sub = 1:n_sub
    
    % just shorter name for easier referencing
    clear d
    d = data_cati{ i_sub }; % d = data 
    clear k
    k = list_cati{ i_sub }; % k = item list 
    
    for i_line = 1:size( d ,1) % n of trials
        % this a rather cumbersome loop through all lines for each participant
        
        
        %% variables from "old" lists, i.e., L = L_load.L_big & k = list_cati
        old_scen = d( i_line, 4); % item/forest idx (according to "old" list of forests, i.e., L
        
        % gains & magnitudes from L (same info in k)
        % d( i_line, 7) contains '7_weather_type', which is coded as 1 or 2
        %   --> thus this variable shifts to the relevant values for the
        %   respective weather type
        d( i_line, 13) = L( old_scen, 4 + d( i_line, 7)); % current g (gain magnitude for environment)
        d( i_line, 14) = L( old_scen, 6 + d( i_line, 7)); % current p (probability for environment)
        d( i_line, 15) = NaN;                             % not used anymore (previoulsy incorrect horizon) therefore set to NaN 
        % correct maximum horizon in column 16
        if which_sample == 1
            d( i_line, 16)   = 5;               % pilot sample always had a maximum horizon of 5 days
        elseif which_sample == 2
            d( i_line, 16)    = k( i_line, 17); % fMRI  sample        had a maximum horizon that was set differently accoding to the list in cati 
        end
        d( i_line, 17) = d( i_line, 16) - d( i_line, 5) + 1; % correct adjusted horizon = current horizon (= days left)
        % snippet from experimental script that shows how this adjusted horizon was set:
        %   cgtext('Tage', 0, +25)
        %   cgtext(num2str( L.list_final(i_forest, 17) - i_gam + 1), 0, -25)
        d( i_line, 18) = k( i_line, 3); % win: PINK YELLOW % IMPORTANT: check win situation: 'win_18'
        % snippet from experimental script that shows this
        %   if L.list_final(i_forest, 3) == 1i_run
        %       col_sur_in = [0,0,0];
        %   elseif L.list_final(i_forest, 3) == 2
        %       col_sur_in = [0.95,0,0.6];
        %   end
        d( i_line, 19) = d( i_line, 6) > d( i_line, 17); % wait-when-save wws; more energy points than days left
        if i_line < size( d ,1) % to keep size similar
            d( i_line+1, 20 ) = d( i_line, 12) - d( i_line, 6); % change in state
        end
        
        %% variables from "new" list, i.e., L_REV policies
        L_REV_horiz   = d( i_line, 17);
        if d( i_line, 7) == 0 % death
            % do nothing
        else
            L_REV_state    = d( i_line, 6) + 1 + ( d( i_line, 7)-1 ) * 6;
            % has to index in a vector of 1x12 with bad and good env plus zero states
            % i_sta+(i_env-1)*n_s
            d( i_line, 21 ) = L_REV.pol_mat( L_REV_horiz,         L_REV_state, old_scen ) - 1; % convert policies to 0 and 1
            d( i_line, 22 ) = L_REV.rew_for( L_REV_horiz     + 1, L_REV_state, old_scen ) - L_REV.rew_wai( L_REV_horiz     + 1, L_REV_state, old_scen );
            d( i_line, 23 ) = L_REV.pol_mat(                   1, L_REV_state, old_scen ) - 1; % convert policies to 0 and 1
            d( i_line, 24 ) = L_REV.rew_for(               1 + 1, L_REV_state, old_scen ) - L_REV.rew_wai(               1 + 1, L_REV_state, old_scen );
        end % if d( i_line, 7) == 0 % death
        
        
        %% calculation of new variables that can be relevant        
        % pure EV
        d( i_line, 25 ) = d( i_line, 13 ) * d( i_line, 14 ) - 2 * (1 - d( i_line, 14 )); % v13: corrected
        
        % log RT
        d( i_line, 26 ) = log( d( i_line, 10 ) );
        
        % binary indication of indifferent optimal policy
        if d( i_line, 22 ) == 0
            d( i_line, 27 ) = 1;
        end

        % binary energy state
        d( i_line, 37 ) = d( i_line, 6 ) == 1;
        
        % proper expected state & difference (v13) adapted from hompred
        % foraging outcome_win
        d( i_line, 38 ) = d( i_line, 6 ) + d( i_line, 13 ); % state plus gain
        if d( i_line, 38 ) > 5
            d( i_line, 38 ) = 5;
        end
   
        % foraging outcome_loss  i.e., foraging can be deadly
        d( i_line, 39 ) = d( i_line, 6 ) - 2; 
        if d( i_line, 39 ) <= 0 % important change <=
            d( i_line, 39 ) = 0;            
        end    
    
        % foraging outcome with probabilities gives expected energy state
        d( i_line, 40 ) = d( i_line, 14) .* d( i_line, 38 ) + ...
            (1-d( i_line, 14)) .* d( i_line, 39 );    
    
        % difference to current state
        d( i_line, 41 ) = d( i_line, 40 ) - d( i_line, 6 ); % put as EV gained
    
        % trials in which foraging could lead to death  
        if d( i_line, 39 ) <= 0 % important change <=        
            d( i_line, 53 ) = 1; % i.e., same as column 39 but reversescored
        end    

        % LD 54_oneIfNrg1_else_pGain
        % take p_gain and set it to 1 in energy state 1
        if d(i_line,6) == 1
            d(i_line,54) = 1;
        else 
            d(i_line,54) = d(i_line,14);
        end     
       
        % LD 55_oneIfNrg1_else_EVenergy
        % take EV and set it to 1 in energy state 1; 
        % this does not make so much sense
        % an appropriate value of this 1-point energy state is difficult to find 
        if d(i_line,6) == 1
            d(i_line,55) = 1;
        else 
            d(i_line,55) = d(i_line,41); % EV in 41
        end 

        % set uppper & lower logical boundaries
        % take p_gain and set it to 1 in energy state 1
        %             and set it to 0 for WWS
        d(i_line,56) = d(i_line,54);         
        if d( i_line, 19) == 1  % WWS in column 19 
            d(i_line,56) = 0;   % 0 indicates that waiting should be done
        end 
       
        
        %% count day & energy combinations
        hist_daen( d(i_line,17), d(i_line,6)+1, i_sub ) = hist_daen( d(i_line,17), d(i_line,6)+1, i_sub ) + 1;
     
    end % for i_line = 1:size( d ,1)
    

    %% split this according to fMRI runs
    % crucially, this split has to be done before deleting the not answered
    % and the "death" trials
    if which_sample == 2
        for i_run = 1:10
            data_pers_fmri{ i_sub, i_run }.d = d( d(:,2) == i_run, : );
            list_cat{ i_sub, i_run }         = list_load{ i_sub, i_run };
            check_54( i_sub, i_run )         = sum( data_pers_fmri{ i_sub, i_run }.d(:,54) == 1 );
        end
    end
    

    %% sizes & choice variability, hist days, energy states
    size_tria( i_sub, 1 ) = sum( d(:, 6) == 0 );
    d( d(:, 6) == 0, :)   = []; % if dead
    size_tria( i_sub, 2 ) = sum( isnan( d(:,9) ) ); % not answered which is in column 9
    d( isnan( d(:,9) ), :)= [];
    size_tria( i_sub, 3 ) = size( d, 1 );     
    size_tria( i_sub, 4 ) = mean( d(:,11) );
    size_tria( i_sub, 5 ) = sum( d(:,19)==1 ); % sum wws
    d_mean(i_sub,:)       = mean( d ); 
    hist_days(i_sub,:)    = hist(d(:,17),1:5);
    hist_ener(i_sub,:)    = hist(d(:, 6),1:5);
    % count rows of final data
    Nr = size(d, 1);
    d_count = [d_count, Nr];
    
    
    %% model-free counting according to rule 3 with WWS in col 56
    d_WWS = [];
    d_WWS = d( d(:,19) == 1, : );
    d_one = [];
    d_one = d( d(:,54) == 1, : );
    for i_rule = 1:rule_n        
        sumi_all_56(i_sub, i_rule)  = sum(     d(:,56) == rule_vect(1,i_rule) );
        sumi_all_54(i_sub, i_rule)  = sum(     d(:,54) == rule_vect(1,i_rule) );
        sumi_WWS_54(i_sub, i_rule)  = sum(     d_WWS(:,54) == rule_vect(1,i_rule) );
        sumi_one_14(i_sub, i_rule)  = sum(     d_one(:,14) == rule_vect(1,i_rule) );
        
        rule_all_54(i_rule,:,i_sub) = mean( d( d(:,54) == rule_vect(1,i_rule), : ) );
        rule_all_56(i_rule,:,i_sub) = mean( d( d(:,56) == rule_vect(1,i_rule), : ) );        
        rule_WWS_54(i_rule,:,i_sub) = mean( d_WWS( d_WWS(:,54) == rule_vect(1,i_rule), : ) );
        rule_one_14(i_rule,:,i_sub) = mean( d_one( d_one(:,14) == rule_vect(1,i_rule), : ) );
    end
    
    
    %% back-naming for using this d below
    data_back{ i_sub } = d;
       
    %% Export concat data
    T = array2table(d)
    T.Properties.VariableNames = genvarname(headers);
    name = join(compose('data_cat/datall_%d', i_sub));
    name2 = [name '.csv'];
    name_fin = strjoin(name2,'');
    writetable(T, string(name_fin));
    
    %% concatenate for R, etc.
    data_R = [data_R; d]; % this is only with relevant behavioral trials
    
    
end % for i_sub


%% Naming of variables
header_columns{1, 1} =  '1_id';
header_columns{1, 2} =  '2_session';
header_columns{1, 3} =  '3_trial';
header_columns{1, 4} =  '4_index_forests';
header_columns{1, 5} =  '5_order_trials_in_forest_(not_days)';
header_columns{1, 6} =  '6_continuous_energy_trial_start';
header_columns{1, 7} =  '7_weather_type';
header_columns{1, 8} =  '8_side_options_on_screen';
header_columns{1, 9} =  '9_button_pressed';
header_columns{1,10} = '10_RT';
header_columns{1,11} = '11_choice';
header_columns{1,12} = '12_continuous_energy_trial_end';
header_columns{1,13} = '13_gain_magnitude';
header_columns{1,14} = '14_p_foraging_gain';
header_columns{1,15} = '15_not_used_set_to_NaN';
header_columns{1,16} = '16_horizon_correct_maximum';
header_columns{1,17} = '17_horizon_correct_adjusted';
header_columns{1,18} = '18_win_magnitude_high_low';
header_columns{1,19} = '19_wait_when_safe';
header_columns{1,20} = '20_energy_change_in_trial';
header_columns{1,21} = '21_optimal_policy_binary';
header_columns{1,22} = '22_optimal_policy';
header_columns{1,23} = '23_pseudo-optimal_horizon-1_binary';
header_columns{1,24} = '24_pseudo-optimal_horizon-1';
header_columns{1,25} = '25_expected_energy_wO_boundaries';
header_columns{1,26} = '26_logRT';
header_columns{1,27} = '27_indifferent_optimal';
for i_hea = 28:36
    header_columns{1,i_hea} = 'empty';
end
header_columns{1,37} = '37_binary_energy';
header_columns{1,38} = '38_foraging_outcome_win';
header_columns{1,39} = '39_foraging_outcome_loss';
header_columns{1,40} = '40_expected_energy_w_boundaries';
header_columns{1,41} = '41_expected_energy_change';
for i_hea = 42:52
    header_columns{1,i_hea} = 'empty';
end
header_columns{1,53} = '53_fatal_foraging_possible';
header_columns{1,54} = '54_oneIfNrg1_else_pGain';
header_columns{1,55} = '55_oneIfNrg1_else_EVenergy';
header_columns{1,56} = '56_rule & p gain 3 WWS';


%% B is the second step for the behavioral data: transformation
% here everything is saved
if which_sample == 1
    save('HOMUNC_data_beh_B_pilot_v1')
elseif which_sample == 2
    save('HOMUNC_data_beh_B_fMRI_v1')
end
% this is the end of the general data transformation