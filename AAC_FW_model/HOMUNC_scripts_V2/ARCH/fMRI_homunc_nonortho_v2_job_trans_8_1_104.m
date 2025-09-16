% from: hstasis_nonortho_v2_job_trans_23_1_2_newpre
%% new getonsets with ortho option
clear all
cwd = pwd;
addpath('/home/sergej/spm12')

%% Change according to your directory strucutre and scan parameters
fs                  = filesep;
dir_base            = cwd; %'/projects/crunchie/korn/HOMUNC_kas';
dir_analysis_base   = 'HOMUNC_ana_fMRI'; % 'HOMUNC_ANALYSIS'; % 
dir_fMRI_base       = 'HOMUNC_data_fMRI_smooth'; % 'DATA_FMRI'; % 'HOMUNC_fMRI'; % 'DATA\FMRI';
dir_scripts         = 'HOMUNC_scripts_v2';
% dir_epi             = ''; % 'functional_1';
% dir_multiple        = 'scanphys_3_multiple_eeg'; % 'scanphys_3_multiple_pulse';

number_of_analysis_behaviour    = '8';
number_of_analysis_scanner      = '1';
number_of_analysis_model        = '104';

ses_string_vec = {'1','2','3','4','5','6','7','8','9','10'};
dir_analysis_behviour           = ['beh_v' number_of_analysis_behaviour];
dir_analysis_scanner            = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model];
cd([dir_base fs dir_analysis_base]);
mkdir(dir_analysis_scanner);

%% load some overall relevant parameters
load(['C:\HAMBURG\HH_DS_2020_30_05_local\HH_ZURICH\HD_HOMUNC\HOMUNC_scripts_v2\beh2fmri_homunc_v' number_of_analysis_behaviour '.mat']);
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
    load([dir_base fs dir_scripts fs dir_analysis_behviour fs filename_sub], 'Z');

    
    cd([dir_base fs dir_analysis_base fs dir_analysis_scanner]);
    dir_analysis_scanner_sub            = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' sub_string];
    mkdir(dir_analysis_scanner_sub);
    
    outputDir = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_analysis_scanner_sub];
    
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
    for k = 1:blockNo
        
        %% scans
        dir_fMRI_sub        = ['HU_smooth_sub_' num2str( sub(n) )];
        k_use = runs{ n_use }(k); % ses_string_vec{k}
        % dir_sess_specified  = ['session_' ses_string_vec{k}];
        dir_sess_specified  = ['session_' num2str(k_use)];
        epiDir              = [dir_base fs dir_fMRI_base fs dir_fMRI_sub fs fs dir_sess_specified];
        f                   = spm_select('List', epiDir, '^swuamr.*\.img$'); % '^swaumr.*\.img$'    % Select smoothed normalised images
        files               = cellstr([repmat([epiDir fs],size(f,1),1) f]);
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).scans                    = files; % '<UNDEFINED>';
        
        f = []; files = [];
        
        %% conditions
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(1).name             = 'forest_1';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(1).onset            = Z.forest{k_use}(:,1)'; % '<UNDEFINED>';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(1).duration         = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(1).tmod             = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(1).orth             = ortho_yn;
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).name             = 'choice_1';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).onset            = Z.choice{k_use}(:,1)'; % '<UNDEFINED>';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).duration         = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).tmod             = 0;
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(1).name     = 'choice_2';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(1).param    = Z.choice{k_use}(:,56)'; % quasit OP, i.e., rule 3 
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(1).poly     = 1;
        
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(2).name     = 'choice_3';
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(2).param    = Z.choice{k_use}(:,62)'; % uncerainty from 22-->60 --> d( i_line, 62 ) = sigmoid_derive( -d( i_line, 60 ) );
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(2).poly     = 1;

       
        
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(3).name     = 'choice_4';
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(3).param    = 
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(3).poly     = 1; 
%         
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(4).name     = 'choice_5';
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(4).param    = 
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(4).poly     = 1; 
%         
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(5).name     = 'choice_6';
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(5).param    = Z.choice{k_use}(:,19)'; % Z.choice{i_ses}( t_trial,11 ) = D.d( t_trial, 10); % RT
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(5).poly     = 1; 
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(2).name     = 'choice_7';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(2).param    = log( Z.choice{k_use}(:,11)' ); % RT
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).pmod(2).poly     = 1; 
        
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(2).orth             = ortho_yn;
        
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(3).name             = 'outcome_1';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(3).onset            = Z.outcome{k_use}(:,1)'; % '<UNDEFINED>';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(3).duration         = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(3).tmod             = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(3).pmod(1).name     = 'outcome_2';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(3).pmod(1).param    = Z.outcome{k_use}(:,2)'; % diff '<UNDEFINED>';
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(3).pmod(1).poly     = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).cond(3).orth             = ortho_yn;
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).multi                    = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).regress                  = struct('name', {}, 'val', {});
        
        
        %% motion, heart, breathing        
%         motion_file     = spm_select('List', epiDir, '^.*\.txt$');
%         multiregFile    = sprintf('regs%d_%d.mat',sub(n),k);        
%         R               = textread([epiDir fs motion_file]);        
%         multiregPath    = [outputDir fs multiregFile];
%         save(multiregPath, 'R');        
%         matlabbatch{1}.spm.stats.fmri_spec.sess(k).multi_reg                = {multiregPath}; % {''}; %  
%         R_read = []; R = [];
        motion_file     = spm_select('List', epiDir, '^.*\.txt$');
        multiregFile    = sprintf('regs%d_%d.mat',sub(n),k_use);        
        R               = textread([epiDir fs motion_file]); 
        %% dummy
%         R(1:3,:)        = [];
        multiregPath    = [outputDir fs multiregFile];
        save(multiregPath, 'R');        
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).multi_reg                = {multiregPath}; % {''}; %  
        R_read = []; R = [];
        
        %% hpf
        matlabbatch{1}.spm.stats.fmri_spec.sess(k).hpf                      = 128;
        
        
    end % block number
    
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