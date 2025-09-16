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
load([dir_base '/AAB_BNW_weather_simple/HOMUNC_scripts_v2/beh2fmri_homunc_v8.mat']);


spm fmri

for n = 1:length(sub) 
    
%     blockNo    = size( Round{ n }.runs, 2); % i.e. sessions
    n_use = find( sub_original==sub(n) );
%     blockNo = size( runs{ n_use }, 2); % i.e. sessions
    blockNo = 1

    sub_string = num2str(sub(n));
    dir_analysis_scanner_sub            = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' sub_string];
  
    T.contrasts = {'forest_1','choice_n1','choice_n2','choice_n3','choice_w1','choice_w2','choice_w3','choice_b1','choice_b2','choice_b3','outcome_1','outcome_2'};
    contrasts = {'ons_boundary', 'mod_boundary', 'rt_boundary', 'mod_allCond', 'RT_allCond', 'ons_N-W','ons_N-B','ons_W-B','mod_N-W','mod_N-B','mod_W-B','lRT_N-W','lRT_N-B','lRT_W-B'};
    contrasts_complex = {'ons_N-WB','ons_W-NB','ons_B-NW','mod_N-WB','mod_W-NB','mod_B-NW','lRT_N-WB','lRT_W-NB','lRT_B-NW'};
%     contrasts_F = {'F_ons_NWB1','F_ons_NWB2','F_mod_NWB1','F_mod_NWB2','F_lRT_NWB1','F_lRT_NWB2'};
%     contrasts_F = {'F_ons_NWB','F_mod_NWB','F_lRT_NWB'};
    
    %% set contrasts
    prepare_con = [eye(size(T.contrasts,2)), zeros(size(T.contrasts,2),length(T.contrasts)) ];
    u = size(prepare_con(1,:))
    
    %% overall effects
    % main effects boundary trials
    c_vec = zeros(1,u(2));
    c_vec(1,5) = 1;
    c_vec(1,8) = 1;
    prepare_con = [prepare_con; c_vec];
    c_vec = zeros(1,u(2));
    c_vec(1,6) = 1;
    c_vec(1,9) = 1;
    prepare_con = [prepare_con; c_vec];
    c_vec = zeros(1,u(2));
    c_vec(1,7) = 1;
    c_vec(1,10) = 1;
    prepare_con = [prepare_con; c_vec];
    % main effect pMods
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = 1;
    c_vec(1,9) = 1;
    prepare_con = [prepare_con; c_vec];
    c_vec = zeros(1,u(2));
    c_vec(1,4) = 1;
    c_vec(1,7) = 1;
    c_vec(1,10) = 1;
    prepare_con = [prepare_con; c_vec];
    
    %% simple contrasts
    % contrast onsets
    c_vec = zeros(1,u(2));
    c_vec(1,2) = 1;
    c_vec(1,5) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,2) = 1;
    c_vec(1,8) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,5) = 1;
    c_vec(1,8) = -1;
    prepare_con = [prepare_con; c_vec];
    
    % contrast model
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,6) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 1;
    c_vec(1,9) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,6) = 1;
    c_vec(1,9) = -1;
    prepare_con = [prepare_con; c_vec];
    
    % contrast RTs
    c_vec = zeros(1,u(2));
    c_vec(1,4) = 1;
    c_vec(1,7) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,4) = 1;
    c_vec(1,10) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,7) = 1;
    c_vec(1,10) = -1;
    prepare_con = [prepare_con; c_vec];
    
    %% complex conntrasts
    % onsets
    c_vec = zeros(1,u(2));
    c_vec(1,2) = 2;
    c_vec(1,5) = -1;
    c_vec(1,8) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,2) = -1;
    c_vec(1,5) = 2;
    c_vec(1,8) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,2) = -1;
    c_vec(1,5) = -1;
    c_vec(1,8) = 2;
    prepare_con = [prepare_con; c_vec];
    
    % model
    c_vec = zeros(1,u(2));
    c_vec(1,3) = 2;
    c_vec(1,6) = -1;
    c_vec(1,9) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,3) = -1;
    c_vec(1,6) = 2;
    c_vec(1,9) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,3) = -1;
    c_vec(1,6) = -1;
    c_vec(1,9) = 2;
    prepare_con = [prepare_con; c_vec];
    
    % RTs
    c_vec = zeros(1,u(2));
    c_vec(1,4) = 2;
    c_vec(1,7) = -1;
    c_vec(1,10) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,4) = -1;
    c_vec(1,7) = 2;
    c_vec(1,10) = -1;
    prepare_con = [prepare_con; c_vec];
    
    c_vec = zeros(1,u(2));
    c_vec(1,4) = -1;
    c_vec(1,7) = -1;
    c_vec(1,10) = 2;
    prepare_con = [prepare_con; c_vec];
    
    % Final contrast matrix
    T.contrasts = [T.contrasts, contrasts];
    T.contrasts = [T.contrasts, contrasts_complex];
%     T.contrasts = [T.contrasts, contrasts_F];
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
