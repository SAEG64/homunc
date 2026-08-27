%% Extract ROI time-series across all factors
% First, save thresholded cluster as .mat in SPM
% Then, export .nii mask from .mat file in marsbar
% Finally, run this code to get the ROI time-series

addpath(genpath('/home/sergej/spm12/toolbox/marsbar'));
savepath;

fs                  = filesep;
path = 'DIRECTORY TO SECOND LEVEL FMRI CONTRASTS FROM FINAL MODEL';
cd(path);

mds = {'choice_nw1_2', 'choice_nw2_2', 'choice_ww1_2', 'choice_ww2_2','choice_bw1_2', 'choice_bw2_2'};

% Define ROI file
% ROIfile = fullfile(path, 'weather_mod_-6_36_-18.nii');
% ROIfile = fullfile(path, 'W-NB_40_14_2.nii');
ROIfile = fullfile(path, 'Fcon_full_44_-34_36.nii');
% ROIfile = fullfile(path, 'N_mod_w1-w2_-4_36_12.nii'); % ACC
% ROIfile = fullfile(path, 'N_mod_w1-w2_-14_-36_38.nii'); % PCC
% ROIfile = fullfile(path, 'N_mod_w1-w2_pos_6_-62_60.nii');
% ROIfile = fullfile(path, 'N_mod_w1-w2_pos_8_-22_-4.nii'); 
% ROIfile = fullfile(path, 'N_mod_w1-w2_pos_38_36_26.nii'); 
% ROIfile = fullfile(path, 'N_mod_w1-w2_pos_48_20_-6.nii'); 
% ROIfile = fullfile(path, 'W_mod_w1-w2_48_-28_40.nii');
% ROIfile = fullfile(path, 'psucc_neg_-42_12_6.nii');

roi_dat = [];
for i = 1:6
    
    % Get SPM design
    md = mds(i);
    md_path = [path fs string(md)];
    spm_mat_file = strjoin([md_path fs 'SPM.mat'], '');
    load(spm_mat_file);
    
    % Get ROI time series
    ROI_ts = Extract_ROI_Data(ROIfile, SPM.xY.P);
%     ROI_ts_2 = spm_summarise(SPM.xY.P, ROIfile, @mean);
    roi_dat = [roi_dat, ROI_ts];
    
end

%% Disable Marsbar to avoid conflict with SPM
rmpath(genpath('/home/sergej/spm12/toolbox/marsbar')); % Remove MarsBaR from path
rehash; % Refresh MATLAB function cache
savepath;

% %% Debug
% % check mask
% % Y = spm_read_vols(spm_vol(ROIfile));
% Y = spm_read_vols(spm_vol(char(SPM.xY.P(1))));
% % disp(['Total voxels: ', num2str(numel(Y))]);
% % disp(['NaN voxels: ', num2str(sum(isnan(Y(:))))]);
% % disp(['Number of ROI voxels: ', num2str(sum(Y(:) > 0))]);
% 
% % check ROI contrast alignment
% roi_mat = spm_vol(char('/media/sergej/Extreme SSD/HOMUNC_parent/HU_FMRI_USE/ZZZZ_SPM/ZZZ_BNW_weather_pSucc_ANOVA/2nd_level_27/mod_interaction/mod_interaction_full_44_-34_36.nii'));
% contrast_mat = spm_vol(char(SPM.xY.P(1)));
% % contrast_mat = spm_vol(char('/media/sergej/Extreme SSD/HOMUNC_parent/HU_FMRI_USE/ZZZZ_SPM/ZZZ_BNW_weather_pSucc_ANOVA/2nd_level_27/mod_interaction/mod_interaction_full_44_-34_36_roi.nii'));
% disp('ROI Affine Matrix:');
% disp(roi_mat);
% disp('Contrast Affine Matrix:');
% disp(contrast_mat);

%% Plotting
% Chose effect type to test
% --------------
% % One-Way Interaction: Weather overall contrasts
% contrast1 = roi_dat(:,1)+roi_dat(:,3)+roi_dat(:,5);      % weather bad
% contrast2 = roi_dat(:,2)+roi_dat(:,4)+roi_dat(:,6);      % weather good
% % Compute mean and standard error (SE) for each contrast
% means = [mean(contrast1), mean(contrast2)];
% std_errors = [std(contrast1) / sqrt(length(contrast1)),
%               std(contrast2) / sqrt(length(contrast2))];

% % One-Way Interaction: N trials weather type
% contrast1 = roi_dat(:,1);      % trade-off weather bad
% contrast2 = roi_dat(:,2);      % trade-off weather good
% % Compute mean and standard error (SE) for each contrast
% means = [mean(contrast1), mean(contrast2)];
% std_errors = [std(contrast1) / sqrt(length(contrast1)),
%               std(contrast2) / sqrt(length(contrast2))];

% % One-Way Interaction: WWS Effect
% contrast1 = roi_dat(:,3) + roi_dat(:,4);      % WWS
% contrast2 = roi_dat(:,1) + roi_dat(:,2) + roi_dat(:,3) + roi_dat(:,4);      % Rest
% % Compute mean and standard error (SE) for each contrast
% means = [mean(contrast1), mean(contrast2)];
% std_errors = [std(contrast1) / sqrt(length(contrast1)),
%               std(contrast2) / sqrt(length(contrast2))];

% Two-Way Interaction Ternary x Weather
contrast1 = roi_dat(:,1);      % trade-off weather bad
contrast2 = roi_dat(:,2);      % trade-off weather good
contrast3 = roi_dat(:,3);      % WWS weather bad
contrast4 = roi_dat(:,4);      % WWS weather good
contrast5 = roi_dat(:,5);      % BES weather bad
contrast6 = roi_dat(:,6);      % BES weather good
% Compute mean and standard error (SE) for each contrast
means = [mean(contrast1), mean(contrast2), mean(contrast3), mean(contrast4), mean(contrast5), mean(contrast6)];
std_errors = [std(contrast1) / sqrt(length(contrast1)),
              std(contrast2) / sqrt(length(contrast2)),
              std(contrast3) / sqrt(length(contrast3)),
              std(contrast4) / sqrt(length(contrast4)),
              std(contrast5) / sqrt(length(contrast5)),
              std(contrast6) / sqrt(length(contrast6))];

% Create bar plot
figure;
bar(means, 'FaceColor', [0.3 0.6 0.9]); % Blue bars

% Add error bars
% --------------
hold on;
% errorbar(1:2, means, std_errors, 'k', 'LineStyle', 'none', 'LineWidth', 2);     % One-Way Interaction
errorbar(1:6, means, std_errors, 'k', 'LineStyle', 'none', 'LineWidth', 2);     % 3x2 interaction
hold off;

% Customize axes
% --------------
% set(gca, 'XTickLabel', {'WWS', 'non-WWS'}); % WWS/non-WWS interaction
% set(gca, 'XTickLabel', {'Bad Weather', 'Good Weather'}); % weather interaction
% set(gca, 'XTickLabel', {'BES Bad Weather', 'BES Good Weather', 'Non-BES Bad Weather', 'Non-BES Good Weather'}); % BESxWeather interaction
set(gca, 'XTickLabel', {'N bad', 'N good', 'W bad', 'W good', 'B bad', 'B good'});  % 3x2 interaction
ylabel('Mean ROI Activation Difference');
title('');
grid on;

% %% Troubleshoot
% for i = 1:length(SPM.xY.P)
%     SPM.xY.P{i} = erase(SPM.xY.P{i}, ',1');  % Remove ',1' from file names
% end

% %% Test significance
% % Test One-Way Interaction with Paired t Test
% roi_values = [contrast1, contrast2]; 
% roi_table = array2table(roi_values, 'VariableNames', {'Contrast1', 'Contrast2'});
% % Define within-subject factor (three levels)
% factor_names = {'Bad Weather', 'Good Weather'};
% % Run Paired t Test
% [h, p, ci, stats] = ttest(contrast1, contrast2);
% fprintf('Paired t test for BOLD difference between weather types: p = %.4f\n', p);

% 3x2 Repeated Measures ANOVA with Interaction
% Build data table
roi_values = [contrast1, contrast2, contrast3, contrast4, contrast5, contrast6];
roi_table = array2table(roi_values, ...
    'VariableNames', {'N_bad', 'N_good', 'W_bad', 'W_good', 'B_bad', 'B_good'});
% Define within-subject factors
FactorA = repmat({'N', 'W', 'B'}, 1, 2)';        % 3 levels
FactorB = [repmat({'bad'}, 3, 1); repmat({'good'}, 3, 1)];  % 2 levels
within_design = table(FactorA, FactorB);
% Fit repeated-measures model
rm = fitrm(roi_table, 'N_bad-B_good ~ 1', 'WithinDesign', within_design);
% Run RM-ANOVA with interaction
ranova_results = ranova(rm, 'WithinModel', 'FactorA*FactorB');
% Display results
disp('3x2 Repeated Measures ANOVA Results with Interaction:');
disp(ranova_results);
% Define all contrasts
contrasts = {contrast1, contrast2, contrast3, contrast4, contrast5, contrast6}; 
% Labels: N_bad, N_good, W_bad, W_good, B_bad, B_good
% -------------------------------
% Simple Interaction 1: contrast12 vs contrast34
% -------------------------------
data_2x2_1 = [contrasts{1}, contrasts{2}, contrasts{3}, contrasts{4}];
tbl_2x2_1 = array2table(data_2x2_1, 'VariableNames', {'N_bad', 'N_good', 'W_bad', 'W_good'});
within_2x2_1 = table(...
    categorical(repmat({'N','N','W','W'}, 1, 1))', ...
    categorical(repmat({'Bad','Good','Bad','Good'}, 1, 1))', ...
    'VariableNames', {'Task','Weather'});
rm_2x2_1 = fitrm(tbl_2x2_1, 'N_bad-W_good ~ 1', 'WithinDesign', within_2x2_1);
ranova_2x2_1 = ranova(rm_2x2_1, 'WithinModel', 'Task*Weather');
disp('Simple 2x2 Interaction (N vs W × Weather):');
disp(ranova_2x2_1);
% -------------------------------
% Simple Interaction 2: contrast12 vs contrast56
% -------------------------------
data_2x2_2 = [contrasts{1}, contrasts{2}, contrasts{5}, contrasts{6}];
tbl_2x2_2 = array2table(data_2x2_2, 'VariableNames', {'N_bad', 'N_good', 'B_bad', 'B_good'});
within_2x2_2 = table(...
    categorical(repmat({'N','N','B','B'}, 1, 1))', ...
    categorical(repmat({'Bad','Good','Bad','Good'}, 1, 1))', ...
    'VariableNames', {'Task','Weather'});
rm_2x2_2 = fitrm(tbl_2x2_2, 'N_bad-B_good ~ 1', 'WithinDesign', within_2x2_2);
ranova_2x2_2 = ranova(rm_2x2_2, 'WithinModel', 'Task*Weather');
disp('Simple 2x2 Interaction (N vs B × Weather):');
disp(ranova_2x2_2);
% -------------------------------
% Simple Interaction 3: contrast34 vs contrast56
% -------------------------------
data_2x2_3 = [contrasts{3}, contrasts{4}, contrasts{5}, contrasts{6}];
tbl_2x2_3 = array2table(data_2x2_3, 'VariableNames', {'W_bad', 'W_good', 'B_bad', 'B_good'});
within_2x2_3 = table(...
    categorical(repmat({'W','W','B','B'}, 1, 1))', ...
    categorical(repmat({'Bad','Good','Bad','Good'}, 1, 1))', ...
    'VariableNames', {'Task','Weather'});
rm_2x2_3 = fitrm(tbl_2x2_3, 'W_bad-B_good ~ 1', 'WithinDesign', within_2x2_3);
ranova_2x2_3 = ranova(rm_2x2_3, 'WithinModel', 'Task*Weather');
disp('Simple 2x2 Interaction (W vs B × Weather):');
disp(ranova_2x2_3);

% Post-hoc comparisons
if ranova_results.pValue(3) < 0.05  % Check interaction effect
    fprintf('Significant interaction (p = %.4f). Running post-hoc comparisons...\n', ranova_results.pValue(3));
    
    % Pairwise combinations (15 total)
    labels = {'N_bad', 'N_good', 'W_bad', 'W_good', 'B_bad', 'B_good'};
    contrasts = {contrast1, contrast2, contrast3, contrast4, contrast5, contrast6};

    n = length(contrasts);
    alpha = 0.05;
    num_comparisons = nchoosek(n, 2);
    alpha_corr = alpha / num_comparisons;  % Bonferroni correction

    fprintf('\nPost-hoc Paired t-Tests (Bonferroni corrected α = %.4f):\n', alpha_corr);

    % Run all pairwise t-tests
    for i = 1:n-1
        for j = i+1:n
            [~, pval, ~, stats] = ttest(contrasts{i}, contrasts{j});
            sig_marker = '';
            if pval < alpha_corr
                sig_marker = '*';
            end
            fprintf('%s vs %s: t(%d) = %.3f, p = %.6f %s\n', ...
                labels{i}, labels{j}, stats.df, stats.tstat, pval, sig_marker);
        end
    end
end
