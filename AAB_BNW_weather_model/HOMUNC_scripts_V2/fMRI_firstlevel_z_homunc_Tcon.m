function firstlevel_z_homunc_1_1_2

%% from: firstlevel_z_homeo_20_1_1_newprephy_new

%% Change according to your scan parameters and directory structure
cwd = '/media/sergej/Extreme SSD/HOMUNC_parent';
cd(cwd);

fs                  = filesep;
% dir_base            = 'C:\HAMBURG\HH_ZURICH\HH_HOMUNC_fMRI'; %'/projects/crunchie/korn/HOMUNC_kas';
% dir_base            = 'C:\HAMBURG\HH_DS_2020_30_05\HH_ZURICH\HH_HOMUNC_fMRI'; %'/projects/crunchie/korn/HOMUNC_kas';
% dir_analysis_base   = 'ANALYSIS'; % 'HOMUNC_ANALYSIS'; % 
dir_base            = cwd; %'/projects/crunchie/korn/HOMUNC_kas';
dir_analysis_base   = 'HU_FMRI_USE'; % 'HOMUNC_ANALYSIS'; % 

number_of_analysis_behaviour    = '12';
number_of_analysis_scanner      = '1';
number_of_analysis_model        = '102';

dir_analysis_scanner            = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model];
sub_original = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333]; 
% sub          = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,    325,326,327,328,334,335]; % 324,
sub          = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];
% sub      = [306,315,319,333]; % [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335]; 
%% load some overall relevant parameters
% load(['C:\HAMBURG\HH_ZURICH\HH_HOMUNC_fMRI\HH_HOMUNC_all\HH_HOMUNC_ana_2\beh2fmri_homunc_v' number_of_analysis_behaviour '.mat']);
% load(['C:\Users\korn\Desktop\stuff\2Do\HD_2020_homunc_manu\HD_2022_homunc_code\beh2fmri_homunc_v' number_of_analysis_behaviour '.mat']);
% Round = R; % renaming because R is used later on for textfiles
% clear R
load([dir_base '/AAB_BNW_weather_model/HOMUNC_scripts_v2/beh2fmri_homunc_v8.mat']);


spm fmri

for n = 1:length(sub) 
    
%     blockNo    = size( Round{ n }.runs, 2); % i.e. sessions
    n_use = find( sub_original==sub(n) );
%     blockNo = size( runs{ n_use }, 2); % i.e. sessions
    blockNo = 1

    sub_string = num2str(sub(n));
    dir_analysis_scanner_sub            = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' sub_string];
  
    T.contrasts = {'forest_1','choice_nw1_1','choice_nw1_2','choice_nw1_3','choice_nw2_1','choice_nw2_2','choice_nw2_3','choice_ww1_1','choice_ww1_2','choice_ww1_3','choice_ww2_1','choice_ww2_2','choice_ww2_3','choice_bw1_1','choice_bw1_2','choice_bw1_3','choice_bw2_1','choice_bw2_2','choice_bw2_3','outcome_1','outcome_2'};
    posthocs = {'W1-W2_ons','W1-W2_RT', 'N-W_ons','N-B_ons','W-B_ons','N-W_mod','N-B_mod','W-B_mod','N-W_rt','N-B_rt','W-B_rt', 'N-(W+B)_ons','W-(N+B)_ons','B-(N+W)_ons','N-(W+B)_mod','W-(N+B)_mod','B-(N+W)_mod','N-(W+B)_rt','W-(N+B)_rt','B-(N+W)_rt', 'NW1-NW2_mod','WW1-WW2_mod','BW1-BW2_mod', '(NW1-NW2)-((WW1-WW2)+(BW1-BW2))_mod', '(NW1-NW2)-(BW1-BW2)_mod', '(NW1-NW2)-(WW1-WW2)_mod', '(BW1-BW2)-(WW1-WW2)_mod'}; % effect factor1, effect factor2, interaction between factor1 and 2
    
    %% set contrasts
    prepare_con = [eye(size(T.contrasts,2)), zeros(size(T.contrasts,2),length(T.contrasts)) ];
    % remove excess columns
    col_n = length(T.contrasts) + 8;
    exces = length(prepare_con)-col_n;
    prepare_con = prepare_con(:, 1:end-exces);
    u = size(prepare_con(1,:))
    
    %% Effect weather type
    % Onsets
    c_vec = zeros(1,u(2));
    c_vec(1,2) = 1;
    c_vec(1,5) = -1;
    c_vec(1,8) = 1;
    c_vec(1,11) = -1;
    c_vec(1,14) = 1;
    c_vec(1,17) = -1;
    prepare_con = [prepare_con; c_vec];
    % RT
    c_vec = zeros(1,u(2));
    c_vec(1,4) = 1;
    c_vec(1,7) = -1;
    c_vec(1,10) = 1;
    c_vec(1,13) = -1;
    c_vec(1,16) = 1;
    c_vec(1,19) = -1;
    prepare_con = [prepare_con; c_vec];
    
    %% Simple effects ternary state
    % Onsets
    % N-W
    c_vec = zeros(1,u(2));
    c_vec(1,2) = 1;
    c_vec(1,5) = 1;
    c_vec(1,8) = -1;
    c_vec(1,11) = -1;
    prepare_con = [prepare_con; c_vec];
    % N-B
    c_vec = zeros(1,u(2));
    c_vec(1,2) = 1;
    c_vec(1,5) = 1;
    c_vec(1,14) = -1;
    c_vec(1,17) = -1;
    prepare_con = [prepare_con; c_vec];
    % W-B
    c_vec = zeros(1,u(2));
    c_vec(1,8) = 1;
    c_vec(1,11) = 1;
    c_vec(1,14) = -1;
    c_vec(1,17) = -1;
    prepare_con = [prepare_con; c_vec];
    % Model
    % N-W
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = 1;
    c_vec(1,9) = -1;
    c_vec(1,12) = -1;
    prepare_con = [prepare_con; c_vec];
    % N-B
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = 1;
    c_vec(1,15) = -1;
    c_vec(1,18) = -1;
    prepare_con = [prepare_con; c_vec];
    % W-B
    c_vec = zeros(1,u(2));
    c_vec(1,9) = 1;
    c_vec(1,12) = 1;
    c_vec(1,15) = -1;
    c_vec(1,18) = -1;
    prepare_con = [prepare_con; c_vec];
    % RT
    % N-W
    c_vec = zeros(1,u(2));
    c_vec(1,4) = 1;
    c_vec(1,7) = 1;
    c_vec(1,10) = -1;
    c_vec(1,13) = -1;
    prepare_con = [prepare_con; c_vec];
    % N-B
    c_vec = zeros(1,u(2));
    c_vec(1,4) = 1;
    c_vec(1,7) = 1;
    c_vec(1,16) = -1;
    c_vec(1,19) = -1;
    prepare_con = [prepare_con; c_vec];
    % W-B
    c_vec = zeros(1,u(2));
    c_vec(1,10) = 1;
    c_vec(1,13) = 1;
    c_vec(1,16) = -1;
    c_vec(1,19) = -1;
    prepare_con = [prepare_con; c_vec];
    
    %% Complex contrast "trade-off" - "boundary" states
    % Onsets: N-(W+B)
    c_vec = zeros(1,u(2));
    c_vec(1,2) = 1;
    c_vec(1,5) = 1;
    c_vec(1,8) = -0.5;
    c_vec(1,11) = -0.5;
    c_vec(1,14) = -0.5;
    c_vec(1,17) = -0.5;
    prepare_con = [prepare_con; c_vec];
    % Onsets: W-(N+B)
    c_vec = zeros(1,u(2));
    c_vec(1,2) = -0.5;
    c_vec(1,5) = -0.5;
    c_vec(1,8) = 1;
    c_vec(1,11) = 1;
    c_vec(1,14) = -0.5;
    c_vec(1,17) = -0.5;
    prepare_con = [prepare_con; c_vec];
    % Onsets: B-(N+W)
    c_vec = zeros(1,u(2));
    c_vec(1,2) = -0.5;
    c_vec(1,5) = -0.5;
    c_vec(1,8) = -0.5;
    c_vec(1,11) = -0.5;
    c_vec(1,14) = 1;
    c_vec(1,17) = 1;
    prepare_con = [prepare_con; c_vec];
    % mod: N-(W+B)
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = 1;
    c_vec(1,9) = -0.5;
    c_vec(1,12) = -0.5;
    c_vec(1,15) = -0.5;
    c_vec(1,18) = -0.5;
    prepare_con = [prepare_con; c_vec];
    % mod: W-(N+B)
    c_vec = zeros(1,u(2));
    c_vec(1,3) = -0.5;
    c_vec(1,6) = -0.5;
    c_vec(1,9) = 1;
    c_vec(1,12) = 1;
    c_vec(1,15) = -0.5;
    c_vec(1,18) = -0.5;
    prepare_con = [prepare_con; c_vec];
    % mod: B-(N+W)
    c_vec = zeros(1,u(2));
    c_vec(1,3) = -0.5;
    c_vec(1,6) = -0.5;
    c_vec(1,9) = -0.5;
    c_vec(1,12) = -0.5;
    c_vec(1,15) = 1;
    c_vec(1,18) = 1;
    prepare_con = [prepare_con; c_vec];
    % rt: N-(W+B)
    c_vec = zeros(1,u(2));
    c_vec(1,4) = 1;
    c_vec(1,7) = 1;
    c_vec(1,10) = -0.5;
    c_vec(1,13) = -0.5;
    c_vec(1,16) = -0.5;
    c_vec(1,19) = -0.5;
    prepare_con = [prepare_con; c_vec];
    % rt: W-(N+B)
    c_vec = zeros(1,u(2));
    c_vec(1,4) = -0.5;
    c_vec(1,7) = -0.5;
    c_vec(1,10) = 1;
    c_vec(1,13) = 1;
    c_vec(1,16) = -0.5;
    c_vec(1,19) = -0.5;
    prepare_con = [prepare_con; c_vec];
    % rt: B-(N+W)
    c_vec = zeros(1,u(2));
    c_vec(1,4) = -0.5;
    c_vec(1,7) = -0.5;
    c_vec(1,10) = -0.5;
    c_vec(1,13) = -0.5;
    c_vec(1,16) = 1;
    c_vec(1,19) = 1;
    prepare_con = [prepare_con; c_vec];
    
    %% Individual weather effects
    % mod
    % NW1-NW2
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = -1;
    prepare_con = [prepare_con; c_vec];
    % WW1-NW2
    c_vec = zeros(1,u(2));
    c_vec(1,9) = 1;
    c_vec(1,12) = -1;
    prepare_con = [prepare_con; c_vec];
    % BW1-BW2
    c_vec = zeros(1,u(2));
    c_vec(1,15) = 1;
    c_vec(1,18) = -1;
    prepare_con = [prepare_con; c_vec];
    
    %% 2x2 interactions
    % mod
    % (NW1-NW2)-(WW1-NW2+BW1-BW2)
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = -1;
    c_vec(1,9) = -0.5;
    c_vec(1,12) = 0.5;
    c_vec(1,15) = -0.5;
    c_vec(1,18) = 0.5;
    prepare_con = [prepare_con; c_vec];
    % (NW1-NW2)-(BW1-BW2)
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = -1;
%     c_vec(1,9) = -0.5;
%     c_vec(1,12) = 0.5;
    c_vec(1,15) = -1;
    c_vec(1,18) = 1;
    prepare_con = [prepare_con; c_vec];
    % (NW1-NW2)-(WW1-WW2)
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = -1;
    c_vec(1,9) = -1;
    c_vec(1,12) = 1;
%     c_vec(1,15) = -1;
%     c_vec(1,18) = 1;
    prepare_con = [prepare_con; c_vec];
    % (BW1-BW2)-(WW1-WW2)
    c_vec = zeros(1,u(2));
%     c_vec(1,3) = 1;
%     c_vec(1,6) = -1;
    c_vec(1,9) = -1;
    c_vec(1,12) = 1;
    c_vec(1,15) = 1;
    c_vec(1,18) = -1;
    prepare_con = [prepare_con; c_vec];
        
    % Final contrast matrix
    T.contrasts = [T.contrasts, posthocs];
    T.contrastVectors = [ repmat(prepare_con, 1, blockNo) ];
    k = 1;
%     jobs{1}.stats{1}.con.spmmat    = {[dir_base fs name_subj{n} fs dir_stats fs 'SPM.mat']};
    jobs{1}.stats{1}.con.spmmat    = {[dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_analysis_scanner_sub fs 'SPM.mat']};
    
    %     Specify tcontrasts
    for cont_nr = 1:length(T.contrasts)
        contrasttype    = T.contrasts{cont_nr};
        contr_input     = T.contrastVectors(cont_nr,:);
        
        % setup job structure for contrasts
        jobs{1}.stats{1}.con.consess{k}.tcon.name           = contrasttype;
        jobs{1}.stats{1}.con.consess{k}.tcon.convec         = contr_input;
        jobs{1}.stats{1}.con.consess{k}.tcon.sessrep        = 'none';
        k = k+1;
    end

    
    % very important little setting. IF 1 THEN ALL EXISTING CONTRASTS ARE
    % DELETED
%     jobs{1}.stats{1}.con.delete                         = 1;

    % outputDir = [dir_base fs name_subj{n} fs dir_stats];
    outputDir = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_analysis_scanner_sub];
    cd (outputDir);
    % save and run job
    save contrasts.mat jobs
    disp(['RUNNING contrast specification for subject number  ' sub_string]);
    spm_jobman('run','contrasts.mat');
    disp(['Contasts created for ' num2str(n) ' subjects']);
    clear jobs

end   % end of subject loop

cd(cwd);
