% from: hstasis_nonortho_v2_job_trans_23_1_2_newpre
%% new getonsets with ortho option
clear all
cwd = '/media/sergej/Extreme SSD/HOMUNC_parent';
addpath('/home/sergej/spm12')

%% Change according to your directory strucutre and scan parameters
fs                  = filesep;
dir_parent            = cwd; %'/projects/crunchie/korn/HOMUNC_kas';
dir_analysis_base   = 'HU_FMRI_USE'; % 'HOMUNC_ANALYSIS'; % 
dir_fMRI_base       = 'HU_FMRI_USE'; % 'DATA_FMRI'; % 'HOMUNC_fMRI'; % 'DATA\FMRI';
dir_scripts         = 'HOMUNC_scripts_v2';
% dir_epi             = ''; % 'functional_1';
% dir_multiple        = 'scanphys_3_multiple_eeg'; % 'scanphys_3_multiple_pulse';

number_of_analysis_behaviour    = '12';
number_of_analysis_scanner      = '1';
number_of_analysis_model        = '102';

ses_string_vec = {'1','2','3','4','5','6','7','8','9','10'};
dir_analysis_behviour           = ['beh_v' number_of_analysis_behaviour '_agg'];
dir_analysis_scanner            = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model];
cd([dir_parent fs dir_analysis_base]);
mkdir(dir_analysis_scanner);

%% load some overall relevant parameters
% load(['C:\HAMBURG\HH_DS_2020_30_05_local\HH_ZURICH\HD_HOMUNC\HOMUNC_scripts_v2\beh2fmri_homunc_v' number_of_analysis_behaviour '.mat']);
load('/media/sergej/Extreme SSD/HOMUNC_parent/AAB_BNW_weather_simple/HOMUNC_scripts_v2/HOMUNC_data_beh_B_fMRI_v1.mat');
ext_header = [{'x57_weather_1_gain_magnitude'}, {'x58_weather_2_gain_magnitude'}, {'x59_weather_1_p_gain'}, {'x60_weather_2_p_gain'}, {'weather_dynamic'}, {'p_delta'}, {'p_delta_dynamic'}, {'BNW_conditions'}, {'BNW_condition'}];
header_columns = [header_columns, ext_header];

% Round = runs; % renaming because R is used later on for textfiles
% clear R

%% subjects
sub_original = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333]; % [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335]; % 
sub          = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];


estimate = 1;
ortho_yn = 0;

%% spm
spm fmri


for n = 1:length(sub)
    
    sub_string = num2str(sub(n));
    
%     blockNo = Round{ n }.runs; % i.e. sessions
    n_use = find( sub_original==sub(n) );
    blockNo = size( runs{ n_use }, 2); % i.e. sessions
   
    filename_sub = ['beh_v' number_of_analysis_behaviour '_sub_' num2str(sub(n)) ];
    clear  'Z'
    load(['/media/sergej/Extreme SSD/HOMUNC_parent/AAB_BNW_weather_simple/HOMUNC_scripts_V2/beh_prep/' fs filename_sub], 'Z');    
    
    cd(['/media/sergej/Extreme SSD/HOMUNC_parent' fs dir_analysis_base fs dir_analysis_scanner]);
    dir_analysis_scanner_sub            = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' sub_string];
    mkdir(dir_analysis_scanner_sub);
    
    outputDir = [dir_parent fs dir_analysis_base fs dir_analysis_scanner fs dir_analysis_scanner_sub];
    
    %-----------------------------------------------------------------------
    % Job saved on 11-Mar-2016 11:20:25 by cfg_util (rev $Rev: 6134 $)
    % spm SPM - SPM12 (6225)
    % cfg_basicio BasicIO - Unknown
    %-----------------------------------------------------------------------
    matlabbatch{1}.spm.stats.fmri_spec.dir              = {outputDir}; % '<UNDEFINED>';
    matlabbatch{1}.spm.stats.fmri_spec.timing.units     = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT        = 2.1;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t    = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0   = 1;
    
    % for k = blockNo
    R = [];
    scans = 0;
    scan_n = [];
    allscans = [];
    ons_forest = [];
    ons_choice_B = [];
    ons_choice_N = [];
    ons_choice_W = [];
    mod_B = [];
    mod_N = [];
    mod_W = [];
    RT_B = [];
    RT_N = [];
    RT_W = [];
    ons_outcome = [];
    outcome = [];
    for k = 1:blockNo
        
        %% get directory strings
        dir_fMRI_sub        = ['HU_sub_' num2str( sub(n) )];
        k_use = runs{ n_use }(k); % ses_string_vec{k};
        % dir_sess_specified  = ['session_' ses_string_vec{k}];
        dir_sess_specified  = ['session_' num2str(k_use)];
        epiDir              = [dir_parent fs dir_fMRI_base fs dir_fMRI_sub fs 'functional_1' fs dir_sess_specified];
        
        %% get scans
        f                   = spm_select('List', epiDir, '^swuamr.*\.img$'); % '^swaumr.*\.img$'    % Select smoothed normalised images
        files               = cellstr([repmat([epiDir fs],size(f,1),1) f]);
        allscans            = [allscans; files]; % '<UNDEFINED>';
        
        %% Regressors
        ons_forest = [ons_forest, Z.forest{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        
        ons_choice_B = [ons_choice_B, Z.choice_B{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        ons_choice_N = [ons_choice_N, Z.choice_N{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        ons_choice_W = [ons_choice_W, Z.choice_W{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        
        RT_B = [RT_B, log( Z.choice_B{k_use}(:,10)')];
        RT_N = [RT_N, log( Z.choice_N{k_use}(:,10)')];
        RT_W = [RT_W, log( Z.choice_W{k_use}(:,10)')];
        
%         % p delta
%         mod_B = [mod_B, Z.choice_B{k_use}(:,62)'];
%         mod_N = [mod_N, Z.choice_N{k_use}(:,62)'];
%         mod_W = [mod_W, Z.choice_W{k_use}(:,62)'];

        % weather as pMod
        mod_B = [mod_B, Z.choice_B{k_use}(:,7)'];
        mod_N = [mod_N, Z.choice_N{k_use}(:,7)'];
        mod_W = [mod_W, Z.choice_W{k_use}(:,7)'];
        
        ons_outcome = [ons_outcome, Z.outcome{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        outcome = [outcome, Z.outcome{k_use}(:,2)'];
        
        
        %% motion artefacts
        extension = 'txt';
        motion_file = dir([epiDir fs '*.' extension]);
        motio     = textread([epiDir fs motion_file.name]); 
        R = [R; motio];
        
        %% scans array
        i_scans = length(files);
        scans = [scans; i_scans];
        scan_n = [scan_n, length(files)]    % count scans per session
        
        %% reset some variables
        f = []; files = [];
        
        
    end
        
    %% conditions
    % Volumes
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans                    = allscans; % '<UNDEFINED>';

    % forest
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).name             = 'forest_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).onset            = ons_forest;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).duration         = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).tmod             = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).orth             = ortho_yn;

    % normal state
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).name             = 'choice_n1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).onset            = ons_choice_N;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).duration         = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).tmod             = 0;
        
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).name     = 'choice_w2';
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).param    = respon_wState; % response
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).poly     = 1;

        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).name     = 'choice_n2';
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).param    = mod_N; % reaction time
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).poly     = 1;
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).name     = 'choice_n3';
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).param    = RT_N; % state posterior
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).poly     = 1;  

    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).orth             = ortho_yn;
    
    % wait when safe state
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).name             = 'choice_w1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).onset            = ons_choice_W;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).duration         = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).tmod             = 0;
        
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).name     = 'choice_f2';
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).param    = respon_fState; % response
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).poly     = 1;

        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).name     = 'choice_w2';
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).param    = mod_W; % reaction time
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).poly     = 1;
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(2).name     = 'choice_w3';
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(2).param    = RT_W; % state posterior
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(2).poly     = 1;  
        
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).orth             = ortho_yn;
    
    % binary energy state
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).name             = 'choice_b1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).onset            = ons_choice_B;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).duration         = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).tmod             = 0;
        
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).name     = 'choice_f2';
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).param    = respon_fState; % response
%         matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).poly     = 1;

        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(1).name     = 'choice_b2';
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(1).param    = mod_B; % reaction time
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(1).poly     = 1;
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(2).name     = 'choice_b3';
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(2).param    = RT_B; % state posterior
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(2).poly     = 1;  

    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).orth             = ortho_yn;

    % outcome
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).name             = 'outcome_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).onset            = ons_outcome; % '<UNDEFINED>';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).duration         = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).tmod             = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(1).name     = 'outcome_2';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(1).param    = outcome; % diff '<UNDEFINED>';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(1).poly     = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).orth             = ortho_yn;

    matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi                    = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).regress                  = struct('name', {}, 'val', {});

    
    %% dummy
    multiregFile    = sprintf('regs%d_%d.mat',sub(n),k_use); 
    multiregPath    = [outputDir fs multiregFile];
    save(multiregPath, 'R');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg                = {multiregPath}; % {''}; %  
    R_read = []; R = [];
    
    %% hpf
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf                      = 128;

    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
    
    %% run batch
    cfg_util('initjob', matlabbatch);
    cfg_util('run', matlabbatch);
    cfg_util('deljob', matlabbatch);
    
    %% concatenate
    spm_fmri_concatenate(fullfile(outputDir, 'SPM.mat'), scan_n);
    
    if estimate
        clear matlabbatch
        outputDir_SPM = [outputDir fs 'SPM.mat'];
        %-----------------------------------------------------------------------
        % Job saved on 11-Mar-2016 12:14:08 by cfg_util (rev $Rev: 6134 $)
        % spm SPM - SPM12 (6225)
        % cfg_basicio BasicIO - Unknown
        %-----------------------------------------------------------------------
        matlabbatch{1}.spm.stats.fmri_est.spmmat = {outputDir_SPM}; % '<UNDEFINED>';
        matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    end
    
    %% run batch
    cfg_util('initjob', matlabbatch);
    cfg_util('run', matlabbatch);
    cfg_util('deljob', matlabbatch);
    
    clear matlabbatch
    
end % subject