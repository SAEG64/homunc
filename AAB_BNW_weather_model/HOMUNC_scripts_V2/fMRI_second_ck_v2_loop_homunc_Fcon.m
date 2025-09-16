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
dir_scripts         = 'AAB_BNW_weather_model/HOMUNC_scripts_v2';

dir_analysis_scanner    = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model];

dir_second_level        = '2nd_level_27';

sub = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];

contrastNames   = {'ons_overall','mod_overall','RT_overall','ons_main_weather','mod_main_weather','RT_main_weather','ons_main_ternary','mod_main_ternary','RT_main_ternary','ons_interaction','mod_interaction','RT_interaction'};
    
conImages       = {'con_0022.nii,1','con_0023.nii,1','con_0024.nii,1','con_0031.nii,1','con_0032.nii,1','con_0033.nii,1',   'con_0025.nii,1','con_0026.nii,1','con_0027.nii,1','con_0028.nii,1','con_0029.nii,1','con_0030.nii,1','con_0034.nii,1','con_0035.nii,1','con_0036.nii,1','con_0037.nii,1','con_0038.nii,1','con_0039.nii,1'};

% single level factors
for i_con = 1:6
    
    con_dirs{i_con} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_second_level fs contrastNames{i_con}];
    mkdir(con_dirs{i_con});
    
    for i_sub = 1:length(sub)
        
        dir_sub = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' num2str( sub( i_sub ) )];
        dir_sub_total = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub];
        
        con_scans{ i_con }.sub{ i_sub, 1 } = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs conImages{ i_con }];
         
    end
end

% multi level factors
c = 1;
con_i = 0;
for i_con = 7:12
    
    con_dirs{i_con} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_second_level fs contrastNames{i_con}];
    mkdir(con_dirs{i_con});
    
    for i_sub = 1:length(sub)
        
        dir_sub = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' num2str( sub( i_sub ) )];
        dir_sub_total = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub];
        
        con_scans1{c}.sub{ i_sub, 1 } = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs conImages{ i_con + con_i }];
        con_scans2{c}.sub{ i_sub, 1 } = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs conImages{ i_con + con_i+1 }];

    end
    
    c = c +1
    con_i = con_i + 1;
    
end


%% original script start here% List of open inputs
% Factorial design specification: Directory - cfg_files
% Factorial design specification: Scans - cfg_files

% single level factors
nrun = 6; % enter the number of runs here
jobfile = {[dir_base fs dir_scripts fs 'second_ck_v2_job_Tcon.m']};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(2, nrun);
for crun = 1:nrun
    inputs{1, crun} = cellstr( con_dirs{ crun }); % Factorial design specification: Directory - cfg_files
    inputs{2, crun} = cellstr( con_scans{ crun }.sub ); % Factorial design specification: Scans - cfg_files
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});


% multi level factors
nrun = (length(conImages)-6)/2; % enter the number of runs here
jobfile = {[dir_base fs dir_scripts fs 'second_ck_v2_job_Fcon.m']};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(2, nrun);
for crun = 1:nrun
    inputs{1, crun} = cellstr( con_dirs{ crun + 6 }); % Factorial design specification: Directory - cfg_files
    inputs{2, crun} = cellstr( con_scans1{ crun }.sub ); % Factorial design specification: Scans - cfg_files
    inputs{3, crun} = cellstr( con_scans2{ crun }.sub ); % Factorial design specification: Scans - cfg_files
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
