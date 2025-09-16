% frmi analysis
% v15: Here conflict between heuristic & optimal model based on
% participant-specific parameter estimates
% FROM: 
% v20: cale_hom_v18_from_v45_difficulty_NN_comment_corr_v1_WITH_32
%% FROM: cale_hom_v23_REV

clear 


%% load data from files
% information about relevant variables in header variable
load('HOMUNC_data_beh_B_fMRI_v1.mat');

number_ana = '12'; % '5A';
name_ana = ['beh_prep'];
mkdir(name_ana)
cd([name_ana]);
ext_header = [{'x57_weather_1_gain_magnitude'}, {'x58_weather_2_gain_magnitude'}, {'x59_weather_1_p_gain'}, {'x60_weather_2_p_gain'}, {'weather_dynamic'}, {'p_delta'}, {'p_delta_dynamic'}, {'BNW_conditions'}, {'BNW_condition'}];
header_columns = [header_columns, ext_header];
beh_dat = '/media/sergej/Extreme SSD/HOMUNC_parent/AAB_BNW_pDel/HOMUNC_scripts_V2/beh_v12_agg';
    
sub_use = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];

for i_sub = 1:length(sub_use)
    
    % get subject ID and data
    s_id = num2str(sub_use(i_sub));
    load([beh_dat fs 'beh_v12_sub' s_id '.mat']);
       
    for i_ses = runs{ i_sub } % 1:size( R{ i_sub }.runs, 2)
        
        D    = data_pers_fmri{ i_sub, i_ses };
        T    = list_cat{ i_sub, i_ses };
        D.c = x(x(:,2) == i_ses,:);
        d_size = size(D.c, 2);
        
        %% from file create_input
        t_c_tim = 1;
        t_trial = 1;
        T.timing_vec_use = [0; T.timing_vec];
        for i = 1:size(T.list_final_sorted, 1)
            
            for j = 1:T.list_final_sorted(i, 17)
                
                
                %% new forest
                if j == 1 % first day in new forest
                    
                    Z.forest{i_ses}( i, 1 ) = T.timing_vec_use(t_c_tim, 1);
                    t_c_tim = t_c_tim + 2;                    
                    Z.forest{i_ses}( i, 2 ) = T.list_final_sorted(i,  2); % state
                    
                end % if j == 1 % first day in new forest
                
                
                %% choice
                Z.choice{i_ses}( t_trial, 1 ) = T.timing_vec_use(t_c_tim, 1); % in that way the subject ID gets overwritten with the time
                t_c_tim = t_c_tim + 1;                
                Z.choice{i_ses}( t_trial, 2:d_size ) = D.c( t_trial, 2:d_size ); % now all variables in the behavioral data list
                % state in col 6 is important for deleting "death"
                
                
                %% choice_RT
                Z.choice_RT{i_ses}( t_trial, 1 ) = Z.choice{i_ses}( t_trial, 1 ) + D.c( t_trial, 10); % i.e. plus RT
                Z.choice_RT{i_ses}( t_trial, 2 ) = D.c( t_trial,  9); % their button press
                
                
                %% outcome
                Z.outcome{i_ses}( t_trial, 1 ) = T.timing_vec_use(t_c_tim, 1);
                t_c_tim = t_c_tim + 2;                
                Z.outcome{i_ses}( t_trial, 2 ) = D.c( t_trial, 12) - D.c( t_trial, 6);             
               
                
                t_trial = t_trial + 1;
                
            end % for j = 1:T.list_final_sorted(i, 17)
            
        end % for i = 1:size(T.list_final_sorted, 1)
        
       
        % delete not_answered
        del_NaN = isnan( Z.choice{i_ses}(:, 9) );Z.choice{i_ses}(:,37) == 0;
        Z.choice{i_ses}(del_NaN, :)     = [];
        Z.outcome{i_ses}(del_NaN, :)    = [];
        Z.choice_RT{i_ses}(del_NaN, :)  = [];
        
        % delete death
        % get zeros from Z.choice state (i.e., col 2)
        % --> delete in Z.choice and Z.outcome
        del_death = Z.choice{i_ses}(:, 6) == 0; % state is in col 6         
        Z.choice{i_ses}(del_death, :)   = [];
        Z.outcome{i_ses}(del_death, :)  = [];
        
%         % filter target range of p success
%         del_high = Z.choice{i_ses}(:, 14) > 0.6;
%         Z.choice{i_ses}(del_high, :)   = [];
%         Z.outcome{i_ses}(del_high, :)  = [];
%         del_low = Z.choice{i_ses}(:, 14) < 0.4;
%         Z.choice{i_ses}(del_low, :)   = [];
%         Z.outcome{i_ses}(del_low, :)  = [];
        
        
        %% v7: now split choice and outcome into two regressors for the two ANSWERS
        del_opt_0 = Z.choice{i_ses}(:,11) == 0;
        del_opt_1 = Z.choice{i_ses}(:,11) == 1;
        
        Z.choice_0{i_ses} = Z.choice{i_ses};
        Z.choice_1{i_ses} = Z.choice{i_ses};        
        Z.choice_0{i_ses}(del_opt_1, :) = [];
        Z.choice_1{i_ses}(del_opt_0, :) = [];
        
        Z.outcome_0{i_ses} = Z.outcome{i_ses};
        Z.outcome_1{i_ses} = Z.outcome{i_ses};        
        Z.outcome_0{i_ses}(del_opt_1, :) = [];
        Z.outcome_1{i_ses}(del_opt_0, :) = [];
        
        %% v9: now split choice and outcome into three regressors for WWS, BES and normal
        del_opt_N = Z.choice{i_ses}(:,19) == 1 | Z.choice{i_ses}(:,37) == 1;
        del_opt_B = Z.choice{i_ses}(:,37) == 0;
        del_opt_W = Z.choice{i_ses}(:,19) == 0;
        
        Z.choice_N{i_ses} = Z.choice{i_ses};
        Z.choice_W{i_ses} = Z.choice{i_ses};        
        Z.choice_B{i_ses} = Z.choice{i_ses};        
        Z.choice_N{i_ses}(del_opt_N, :) = [];
        Z.choice_W{i_ses}(del_opt_W, :) = [];
        Z.choice_B{i_ses}(del_opt_B, :) = [];
        
        Z.outcome_N{i_ses} = Z.outcome{i_ses};
        Z.outcome_W{i_ses} = Z.outcome{i_ses};        
        Z.outcome_B{i_ses} = Z.outcome{i_ses};        
        Z.outcome_N{i_ses}(del_opt_N, :) = [];
        Z.outcome_W{i_ses}(del_opt_W, :) = [];
        Z.outcome_B{i_ses}(del_opt_B, :) = [];
        
        %% v10: now split choices and regressors into two regressors for weather type
        del_opt_N_F = Z.choice_N{i_ses}(:,11) == 0
        del_opt_N_W = Z.choice_N{i_ses}(:,11) == 1
        del_opt_B_F = Z.choice_B{i_ses}(:,11) == 0
        del_opt_B_W = Z.choice_B{i_ses}(:,11) == 1
        del_opt_W_F = Z.choice_W{i_ses}(:,11) == 0
        del_opt_W_W = Z.choice_W{i_ses}(:,11) == 1
        
        % choices
        Z.choice_N_F{i_ses} = Z.choice_N{i_ses};
        Z.choice_N_W{i_ses} = Z.choice_N{i_ses};
        
        Z.choice_B_F{i_ses} = Z.choice_B{i_ses};        
        Z.choice_B_W{i_ses} = Z.choice_B{i_ses};        

        Z.choice_W_F{i_ses} = Z.choice_W{i_ses}; 
        Z.choice_W_W{i_ses} = Z.choice_W{i_ses}; 
        
        Z.choice_N_F{i_ses}(del_opt_N_F, :) = [];
        Z.choice_N_W{i_ses}(del_opt_N_W, :) = [];

        Z.choice_B_F{i_ses}(del_opt_B_F, :) = [];
        Z.choice_B_W{i_ses}(del_opt_B_W, :) = [];
        
        Z.choice_W_F{i_ses}(del_opt_W_F, :) = [];
        Z.choice_W_W{i_ses}(del_opt_W_W, :) = [];
        
        % outcomes
        Z.outcome_N_F{i_ses} = Z.outcome_N{i_ses};
        Z.outcome_N_W{i_ses} = Z.outcome_N{i_ses};
        
        Z.outcome_B_F{i_ses} = Z.outcome_B{i_ses};        
        Z.outcome_B_W{i_ses} = Z.outcome_B{i_ses};        

        Z.outcome_W_F{i_ses} = Z.outcome_W{i_ses}; 
        Z.outcome_W_W{i_ses} = Z.outcome_W{i_ses}; 
        
        Z.outcome_N_F{i_ses}(del_opt_N_F, :) = [];
        Z.outcome_N_W{i_ses}(del_opt_N_W, :) = [];

        Z.outcome_B_F{i_ses}(del_opt_B_F, :) = [];
        Z.outcome_B_W{i_ses}(del_opt_B_W, :) = [];
        
        Z.outcome_W_F{i_ses}(del_opt_W_F, :) = [];
        Z.outcome_W_W{i_ses}(del_opt_W_W, :) = [];
        
        clear D T       
        
    end % for i_ses = 1:size( R{ sub_vec_incl( i_sub ) }.info, 2)
   
    save(['beh_v' number_ana '_sub_' num2str( sub_use( i_sub ) )], 'Z')
    
    clear Z
    
end % for i_sub = 1:length(sub_vec_incl)