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

contrastNames   = {'forest_1','choice_nw1_1','choice_nw1_2','choice_nw1_3','choice_nw2_1','choice_nw2_2','choice_nw2_3','choice_ww1_1','choice_ww1_2','choice_ww1_3','choice_ww2_1','choice_ww2_2','choice_ww2_3','choice_bw1_1','choice_bw1_2','choice_bw1_3','choice_bw2_1','choice_bw2_2','choice_bw2_3','outcome_1','outcome_2','W1-W2_ons','W1-W2_RT', 'N-W_ons','N-B_ons','W-B_ons','N-W_mod','N-B_mod','W-B_mod','N-W_rt','N-B_rt','W-B_rt', 'N-(W+B)_ons','W-(N+B)_ons','B-(N+W)_ons','N-(W+B)_mod','W-(N+B)_mod','B-(N+W)_mod','N-(W+B)_rt','W-(N+B)_rt','B-(N+W)_rt', 'NW1-NW2_mod','WW1-WW2_mod','BW1-BW2_mod', '(NW1-NW2)-((WW1-WW2)+(BW1-BW2))_mod', '(NW1-NW2)-(BW1-BW2)_mod', '(NW1-NW2)-(WW1-WW2)_mod', '(BW1-BW2)-(WW1-WW2)_mod'}; % regressor effects, effect factor1, effect factor2, interaction between factor1 and 2
    
conImages       = {'con_0001.nii,1','con_0002.nii,1','con_0003.nii,1','con_0004.nii,1','con_0005.nii,1','con_0006.nii,1','con_0007.nii,1','con_0008.nii,1','con_0009.nii,1','con_0010.nii,1','con_0011.nii,1','con_0012.nii,1','con_0013.nii,1','con_0014.nii,1','con_0015.nii,1','con_0016.nii,1','con_0017.nii,1','con_0018.nii,1','con_0019.nii,1','con_0020.nii,1','con_0021.nii,1','con_0022.nii,1','con_0023.nii,1','con_0024.nii,1','con_0025.nii,1','con_0026.nii,1','con_0027.nii,1','con_0028.nii,1','con_0029.nii,1','con_0030.nii,1','con_0031.nii,1','con_0032.nii,1','con_0033.nii,1','con_0034.nii,1','con_0035.nii,1','con_0036.nii,1','con_0037.nii,1','con_0038.nii,1','con_0039.nii,1','con_0040.nii,1','con_0041.nii,1','con_0042.nii,1','con_0043.nii,1','con_0044.nii,1','con_0045.nii,1','con_0046.nii,1','con_0047.nii,1','con_0048.nii,1'};

for i_con = 1:length(contrastNames)
    
    con_dirs{i_con} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_second_level fs contrastNames{i_con}];
    mkdir(con_dirs{i_con});
    
    for i_sub = 1:length(sub)
        
        dir_sub = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' num2str( sub( i_sub ) )];
        dir_sub_total = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub];
        
        con_scans{ i_con }.sub{ i_sub, 1 } = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs conImages{ i_con }];
         
    end
end


%% original script start here% List of open inputs
% Factorial design specification: Directory - cfg_files
% Factorial design specification: Scans - cfg_files
nrun = length(contrastNames); % enter the number of runs here
jobfile = {[dir_base fs dir_scripts fs 'second_ck_v2_job_Tcon.m']};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(2, nrun);
for crun = 1:nrun
    inputs{1, crun} = cellstr( con_dirs{ crun }); % Factorial design specification: Directory - cfg_files
    inputs{2, crun} = cellstr( con_scans{ crun }.sub ); % Factorial design specification: Scans - cfg_files
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
