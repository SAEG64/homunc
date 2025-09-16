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
load('/media/sergej/Extreme SSD/HOMUNC_parent/AAA_single_pMod/HOMUNC_scripts_v2/HOMUNC_data_beh_B_fMRI_v1.mat');
ext_header = [{'57_weather_1_gain_magnitude'}, {'58_weather_2_gain_magnitude'}, {'59_weather_1_p_gain'}, {'60_weather_2_p_gain'}, {'eather_dynamic'}, {'_delta'}, {'_delta_dynamic'}, {'NW_conditions'}, {'_delta_BNW_conditions_fit'}, {'_delta_BNW_conditions_uncertain'}, {'_delta_BNW_conditions_uncertain_fit'}, {'action_net'}, {'value_net'}, {'pi_net_z'}];
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
    load(['/media/sergej/Extreme SSD/HOMUNC_parent/AAA_single_pMod/HOMUNC_scripts_V2/beh_prep/' fs filename_sub], 'Z');    
    
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
    ons_choice = [];
    mod = [];
    RT = [];
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
        
        ons_choice = [ons_choice, Z.choice{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        
        % RT
        RT = [RT, log( Z.choice{k_use}(:,10)')];

        % responses
%         logits = Z.choice{k_use}(:,70)'; % z-scored action values
%         probs = 1 ./ (1 + exp(-logits));
%         mod = [mod, probs];
%         tol = 0.01;
%         mod(mod > 0.5 + tol) = 1;  % Action F
%         mod(mod < 0.5 - tol) = 0;  % Action W
%         mod(abs(mod - 0.5) <= tol) = 0.5;  % Indifferent
%         mod = [mod, Z.choice{k_use}(:,22)']; % optimal policy
%         mod = [mod, Z.choice{k_use}(:,68)']; % action net            
%         mod = [mod, Z.choice{k_use}(:,69)']; % value net
        p = Z.choice{k_use}(:,68);
        % Clip p to avoid values too close to 0 or 1
        p = min(max(p, 1e-10), 1 - 1e-10);  % Clip to [1e-10, 1 - 1e-10]
        entropy = - (p .* log2(p) + (1 - p) .* log2(1 - p));
        mod = [mod, entropy']; % action net entropy
        
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
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).name             = 'choice_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).onset            = ons_choice;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).duration         = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).tmod             = 0;

        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).name     = 'choice_2';
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).param    = mod; % reaction time
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).poly     = 1;
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).name     = 'choice_3';
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).param    = RT; % state posterior
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).poly     = 1;  

    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).orth             = ortho_yn;

    % outcome
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).name             = 'outcome_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).onset            = ons_outcome; % '<UNDEFINED>';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).duration         = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).tmod             = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).name     = 'outcome_2';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).param    = outcome; % diff '<UNDEFINED>';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).poly     = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).orth             = ortho_yn;

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