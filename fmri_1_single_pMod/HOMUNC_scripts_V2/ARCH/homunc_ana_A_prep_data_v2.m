%% HOMeostasis UNCertainty
% this script PREProcesses the data
% v23 clean up 2023.09.20 by christoph.w.korn@gmail.com
% behavioral analyses & preparation of fMRI setup
% based on new simplified folder structure
clear
close all


%% directories & selection/deletion options
fs              = filesep;
which_sample    = 2; % 1=behavioral sample; 2=fMRI sample
dir_base        = 'C:\HAMBURG\HH_DS_2020_30_05_local\HH_ZURICH\HD_HOMUNC';
dir_scripts     = 'HOMUNC_scripts_v2';


%% samples with directories
if which_sample == 1
    dir_behave  = 'HOMUNC_data_beh_pilot';
    sub_use     = [101:107,110:120,122:131]; 
    % exclude 108,109 because they had participated in hstasis 
    % exclude 121 due to 85% criterion
elseif which_sample == 2
    dir_behave  = 'HOMUNC_data_beh_fMRI';
    sub_use     = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];
    % Info from Notes for fMRI sample
    % 302 sessions 6, 7 do not exist because of real-time error
    % 304 2x real-time control error in session 4 but restarted
    % 306 T1 missing
    % 308 training not completed
    % 310 real-time control error in session 3, not redone
    % 312 session 6 has artefacts (in MPFC, etc.) only this session excluded
    % 313 only 9 sessions, time constraints
    % 315 coregistration initially failed, now good
    % 317 real-time error session 6 but restarted
    % 319 coregistration initially failed, now good
    % 324 session 4 does not exist; only 9 sessions, presed alarm for no reason; manually changed numbers of sessions on cmre01!
    % 326 session 6 missing
    % 327 session 9 restarted
    % 328 sessions 9, 10 missing
    % 333 session 10 missing in fMRI data, T1 missing
end


%% put structure for runs with specific runs for fMRI subjects
n_sub   = length(sub_use);
runs    = cell( n_sub, 1 );
if which_sample == 1
    for i_sub = 1:n_sub
        run_max = 5;
        runs{ i_sub } = 1:run_max;
        % pilot sample: 5 runs per subject; no missing runs
    end
elseif which_sample == 2
    % fMRI sample: 10 runs per subjects; missing runs for some subjects
    % (see infos above)
    run_max = 10;
    for i_sub = 1:n_sub
        if sub_use(i_sub) == 302
            runs{ i_sub } = [1:5,8:10];
        elseif sub_use(i_sub) == 310
            runs{ i_sub } = [1:2,4:10];
        elseif sub_use(i_sub) == 312
            runs{ i_sub } = [1:5,7:10];
        elseif sub_use(i_sub) == 313
            runs{ i_sub } = 1:9;
        elseif sub_use(i_sub) == 324
            runs{ i_sub } = [1:3,5:10];
        elseif sub_use(i_sub) == 326
            runs{ i_sub } = [1:5,7:10];
        elseif sub_use(i_sub) == 328
            runs{ i_sub } = 1:8;
        elseif sub_use(i_sub) == 333
            runs{ i_sub } = 1:9;
        else
            runs{ i_sub } = 1:run_max;
        end        
    end
end


%% load and preprocess behavioral data
% i.e., load DATA % ITEM LIST
% concatenate over all runs 
% result: data_cati & list_cati
% finally, list_cati is only used sparely in the analyses below
data_stru   = cell( n_sub, run_max );   % STRUcture
list_expa   = cell( n_sub, run_max );   % EXPAnde
data_cati   = cell( n_sub, 1 );         % catiru conCATanate over I_RUn
list_cati   = cell( n_sub, 1 );         
check_size  = NaN(  n_sub, 1 );

for i_sub = 1:n_sub   
    
    for i_run = 1:size( runs{ i_sub }, 2 )    % number of runs for subjects
        
        run_use = runs{ i_sub }( i_run );     % get specific run numbers for subject  
        if which_sample == 1                    % naming of files slightly different for pilot & fMRI
            file_name = ['sub_' num2str( sub_use(i_sub) ) '_dyn_train_2_' num2str( run_use )];
        elseif which_sample == 2
            file_name = ['sub_' num2str( sub_use(i_sub) ) '_HOMUNC_r_' num2str( run_use )];
        end
        
        %% get DATA current run i.e., load the file with all behavioral data, 
        data_load(i_sub, run_use) = load([dir_base fs dir_behave fs file_name]);    % structure not preallocated
        % extract the relevant data saved in "DATA.exp_data"
        % data_stru(:,1) = sub number
        %          (:,2) = run number (1:10)
        %          (:,3:13) = data from data file: according to the following list
        % DATA.exp_data(t_count,1) = t_count;
        % DATA.exp_data(t_count,2) = L.list_final(i_forest, 1); % scenario_new
        % DATA.exp_data(t_count,3) = i_gam;
        % DATA.exp_data(t_count,4) = li_gam; % li_gam = L.list_final(i_forest, 2); % start state
        % DATA.exp_data(t_count,5) = env;
        % DATA.exp_data(t_count,6) = rand_side_hunt(1);
        % DATA.exp_data(t_count,7) = k;
        % DATA.exp_data(t_count,8) = after - before;
        % DATA.exp_data(t_count,9) = choice;
        % DATA.exp_data(t_count,10) = li_gam;
        data_size = size( data_load(i_sub, run_use).DATA.exp_data, 1 );             % n of trials
        data_stru{ i_sub, run_use } = [zeros(data_size,1) + sub_use(i_sub), ...     % subject label
            zeros(data_size,1) + run_use, ...                                       % run label!
            data_load(i_sub, run_use).DATA.exp_data];
        
        %% get ITEM LIST current run  
        % match size to data size (repeat item line for n of days of item/forest)
        if which_sample == 1
            % relevant item list in: L.list_final of loaded data
            list_expa{ i_sub, run_use } = [];
            for i_trial = 1:size( data_load(i_sub, run_use).L.list_final, 1 )
                clear line_add
                line_add = repmat( data_load(i_sub, run_use).L.list_final(i_trial,:), 5, 1 );
                list_expa{ i_sub, run_use } = [list_expa{ i_sub, run_use }; line_add];
            end
        elseif which_sample == 2
            % load list of current run, which depends on the pseudorandomization of the lists
            list_name = ['list_HOMUNC_list_' ...
                num2str(data_load(i_sub, run_use).list_take_2(1,data_load(i_sub, run_use).run_num)) '_v1'];            
            list_load{ i_sub, run_use } = load([dir_base fs dir_behave fs list_name]); % structure not preallocated
            list_expa{ i_sub, run_use } = [];
            for i_trial = 1:size( list_load{ i_sub, run_use }.list_final_sorted, 1 )
                clear line_add
                % repeat item(=forest) line for n of days of current forest 
                line_add = repmat( list_load{ i_sub, run_use }.list_final_sorted(i_trial,:), list_load{ i_sub, run_use }.list_final_sorted(i_trial,17), 1 );
                list_expa{ i_sub, run_use } = [list_expa{ i_sub, run_use }; line_add];
            end
        end
        
        %% concatenate over all runs
        data_cati{ i_sub } = [data_cati{ i_sub }; data_stru{ i_sub, run_use }];
        list_cati{ i_sub } = [list_cati{ i_sub }; list_expa{ i_sub, run_use }];
        
    end
    
    check_size(i_sub,1) = size( data_cati{ i_sub }, 1 ) - size( list_cati{ i_sub }, 1 );
    
end
cd([dir_base fs dir_scripts])
% A is the first step for the behavioral data: preprocessing 
if which_sample == 1
    save('HOMUNC_data_beh_A_pilot_v1','data_cati','list_cati','sub_use')
elseif which_sample == 2
    save('HOMUNC_data_beh_A_fMRI_v1', 'data_cati','list_cati','sub_use','runs','list_load')
end
% this is the end of the general data loading & preparation