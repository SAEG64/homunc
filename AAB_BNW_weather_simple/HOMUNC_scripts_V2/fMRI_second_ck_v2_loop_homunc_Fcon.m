% from: second_ck_v2_loop_25_1_3_newprephy_26_new_ADD

clear all

spm_jobman('initcfg');

fs = filesep;
number_of_analysis_behaviour    = '12';
number_of_analysis_scanner      = '1';
number_of_analysis_model        = '102';

% dir_base            = 'C:\HAMBURG\HH_DS_2020_30_05\HH_ZURICH\HH_HOMUNC_fMRI'; %'C:\HAMBURG\HH_ZURICH\HH_HOMUNC_fMRI'; % 'H:\HH_ZURICH_1'; %  'H:\HH_ZURICH_1'; %
% dir_analysis_base   = 'ANALYSIS'; % 'ANALYSIS';

dir_base            = '/media/sergej/Extreme SSD/HOMUNC_parent'; %'/projects/crunchie/korn/HOMUNC_kas';
dir_analysis_base   = 'HU_FMRI_USE'; % 'HOMUNC_ANALYSIS'; % 
dir_scripts         = 'AAB_BNW_weather_simple/HOMUNC_scripts_v2';

dir_analysis_scanner    = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model];

dir_second_level        = '2nd_level_27';

sub = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];

contrastNames   = {'F_ons_NWB','F_mod_NWB','F_lRT_NWB'};
    
conImages       = {'con_0001.nii,1','con_0002.nii,1','con_0003.nii,1','con_0004.nii,1','con_0005.nii,1','con_0006.nii,1','con_0007.nii,1','con_0008.nii,1','con_0009.nii,1','con_0010.nii,1','con_0011.nii,1','con_0012.nii,1','con_0013.nii,1','con_0014.nii,1','con_0015.nii,1','con_0016.nii,1','con_0017.nii,1','con_0018.nii,1'};
con_i = 1;
w_enc = 0;
for i_con = 13:13+length(contrastNames)-1
    
    con_dirs{con_i} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_second_level fs contrastNames{con_i}];
    mkdir(con_dirs{con_i});
    
    for i_sub = 1:length(sub)
        
        dir_sub = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' num2str( sub( i_sub ) )];
        dir_sub_total = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub];
        
        con_scans1{ con_i }.sub{ i_sub, 1 } = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs conImages{ i_con + w_enc }];
        con_scans2{ con_i }.sub{ i_sub, 1 } = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs conImages{ i_con + w_enc+1 }];

    end
    
    con_i = con_i + 1;
    w_enc = w_enc + 1;
    
end


%% original script start here% List of open inputs
% Factorial design specification: Directory - cfg_files
% Factorial design specification: Scans - cfg_files
nrun = length(contrastNames); % enter the number of runs here
jobfile = {[dir_base fs dir_scripts fs 'second_ck_v2_job_Fcon.m']};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(2, nrun);
for crun = 1:nrun
    inputs{1, crun} = cellstr( con_dirs{ crun }); % Factorial design specification: Directory - cfg_files
    inputs{2, crun} = cellstr( con_scans1{ crun }.sub ); % Factorial design specification: Scans - cfg_files
    inputs{3, crun} = cellstr( con_scans2{ crun }.sub ); % Factorial design specification: Scans - cfg_files
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
