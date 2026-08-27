% ===========================
% Second level (group) models
% ===========================
clear all
spm_jobman('initcfg');

fs = filesep;
number_of_analysis_behaviour = '12';
number_of_analysis_scanner   = '1';
number_of_analysis_model     = '102';

dir_base          = 'DIRECTORY TO FMRI DATA: HOMUNC_parent';
dir_analysis_base = 'HU_FMRI_USE';
dir_scripts       = 'fmri_2_BNW_weather_model_banded/HOMUNC_scripts_v2';
dir_analysis_scanner = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model];

% Output root for second level
dir_second_level  = '2nd_level_models';

% Subjects
sub = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];

% -------- Target 2nd-level model names (exactly as you asked)
targetNames_T = { ...
  'ons_overall', ...
  'mod_overall', ...          % use pin+pout (see resolution below)
  'RT_overall', ...
  'ons_main_weather', ...
  'mod_main_weather', ...     % pin only
  'RT_main_weather' ...
};

targetNames_F = { ...
  'ons_main_ternary', ...
  'mod_main_ternary', ...     % pin only
  'RT_main_ternary', ...
  'ons_interaction', ...
  'mod_interaction', ...      % pin only
  'RT_interaction' ...
};

% ---- Map each 2nd-level model to 1st-level contrast names
% One-sample models (single contrast per subject)
% Note: "mod_overall" tries 'mod_overall'; if absent, will be moved to paired section as pin_overall+pout_overall
map_T = containers.Map;
map_T('ons_overall')        = {'ons_overall'};
map_T('mod_overall')        = {'mod_overall'};          % fallback handled later
map_T('RT_overall')         = {'RT_overall'};

map_T('ons_main_weather')   = {'ons_main_weather'};
map_T('mod_main_weather')   = {'pin_main_weather'};     % pin only (your request)
map_T('RT_main_weather')    = {'RT_main_weather'};

% Paired models (two contrasts per subject)
map_F = containers.Map;
% main ternary: pair N>W and W>B
map_F('ons_main_ternary')   = {'ons_main_ternary_N_gt_W','ons_main_ternary_W_gt_B'};
map_F('mod_main_ternary')   = {'pin_main_ternary_N_gt_W','pin_main_ternary_W_gt_B'};   % pin only
map_F('RT_main_ternary')    = {'RT_main_ternary_N_gt_W','RT_main_ternary_W_gt_B'};

% interactions: pair type 1 and type 2
map_F('ons_interaction')    = {'ons_interaction_1','ons_interaction_2'};
map_F('mod_interaction')    = {'pin_interaction_1','pin_interaction_2'};               % pin only
map_F('RT_interaction')     = {'RT_interaction_1','RT_interaction_2'};

% Utility: find 1st-level con number in a subject by contrast name
find_con_num = @(spm, name) find(strcmp({spm.xCon.name}, name), 1, 'first');

% Containers for job inputs
t_dirs = cell(1, numel(targetNames_T));
t_scans = cell(1, numel(targetNames_T));   % each is a cellstr list of con files

f_dirs = cell(1, numel(targetNames_F));
f_scans1 = cell(1, numel(targetNames_F));  % first condition
f_scans2 = cell(1, numel(targetNames_F));  % second condition

% Prepare directories
for i = 1:numel(targetNames_T)
    t_dirs{i} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_second_level fs targetNames_T{i}];
    if ~exist(t_dirs{i}, 'dir'), mkdir(t_dirs{i}); end
end
for i = 1:numel(targetNames_F)
    f_dirs{i} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_second_level fs targetNames_F{i}];
    if ~exist(f_dirs{i}, 'dir'), mkdir(f_dirs{i}); end
end

% Resolve con files per subject
% Also track if we need to *promote* mod_overall to paired (pin_overall + pout_overall)
use_mod_overall_as_paired = false;
tmp_mod_overall_scans_single = cell(numel(sub),1); % if available as single

for i_con = 1:numel(targetNames_T)
    t_scans{i_con} = cell(numel(sub),1);
end
for i_con = 1:numel(targetNames_F)
    f_scans1{i_con} = cell(numel(sub),1);
    f_scans2{i_con} = cell(numel(sub),1);
end

for i_sub = 1:numel(sub)
    subj = sub(i_sub);
    dir_sub = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' num2str(subj)];
    spm_path = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs 'SPM.mat'];
    if ~exist(spm_path, 'file')
        error('SPM.mat not found for subject %d at %s', subj, spm_path);
    end
    S = load(spm_path, 'SPM'); SPM = S.SPM;

    % ---- Fill T (one-sample) by name
    for i = 1:numel(targetNames_T)
        model_name = targetNames_T{i};
        need = map_T(model_name);  % 1 name (cell)
        name1 = need{1};

        if strcmp(model_name, 'mod_overall')
            % Try 'mod_overall' first
            idx = find_con_num(SPM, 'mod_overall');
            if ~isempty(idx)
                confile = sprintf('con_%04d.nii,1', idx);
                t_scans{i}{i_sub,1} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs confile];
                tmp_mod_overall_scans_single{i_sub} = t_scans{i}{i_sub,1};
            else
                % fallback to paired: pin_overall + pout_overall
                use_mod_overall_as_paired = true;
            end
        else
            idx = find_con_num(SPM, name1);
            if isempty(idx)
                error('Missing 1st-level contrast "%s" for subject %d', name1, subj);
            end
            confile = sprintf('con_%04d.nii,1', idx);
            t_scans{i}{i_sub,1} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs confile];
        end
    end

    % ---- Fill F (paired) by name (always two)
    for i = 1:numel(targetNames_F)
        model_name = targetNames_F{i};
        need = map_F(model_name);
        nameA = need{1}; nameB = need{2};

        idxA = find_con_num(SPM, nameA);
        idxB = find_con_num(SPM, nameB);
        if isempty(idxA) || isempty(idxB)
            error('Missing 1st-level contrasts "%s" and/or "%s" for subject %d', nameA, nameB, subj);
        end
        confA = sprintf('con_%04d.nii,1', idxA);
        confB = sprintf('con_%04d.nii,1', idxB);

        f_scans1{i}{i_sub,1} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs confA];
        f_scans2{i}{i_sub,1} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs confB];
    end

    % If fallback needed, also collect pin_overall+pout_overall now
    if use_mod_overall_as_paired
        idxPin  = find_con_num(SPM, 'pin_overall');
        idxPout = find_con_num(SPM, 'pout_overall');
        if isempty(idxPin) || isempty(idxPout)
            error('Fallback for mod_overall requires pin_overall and pout_overall for subject %d', subj);
        end
        confPin  = sprintf('con_%04d.nii,1', idxPin);
        confPout = sprintf('con_%04d.nii,1', idxPout);
        % We will insert this as an extra paired model with name 'mod_overall'
        % (Handled below in job launch section)
        mod_overall_pair_scans1{i_sub,1} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs confPin];
        mod_overall_pair_scans2{i_sub,1} = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_sub fs confPout];
    end
end

% =========================
% Launch one-sample designs
% =========================
% If mod_overall was resolved via fallback, skip it here and run as paired later.
jobfile_T = {[dir_base fs dir_scripts fs 'second_ck_v2_job_Tcon.m']};
keep_idx = true(1, numel(targetNames_T));
if use_mod_overall_as_paired
    keep_idx(strcmp(targetNames_T,'mod_overall')) = false;
end

nrun_T = sum(keep_idx);
if nrun_T > 0
    jobs = repmat(jobfile_T, 1, nrun_T);
    inputs = cell(2, nrun_T);
    wr = 1;
    for i = 1:numel(targetNames_T)
        if ~keep_idx(i), continue; end
        inputs{1, wr} = cellstr( t_dirs{i} );        % Directory
        inputs{2, wr} = cellstr( t_scans{i} );       % Scans
        wr = wr + 1;
    end
    spm('defaults','fMRI');
    spm_jobman('run', jobs, inputs{:});
end

% ======================
% Launch paired designs
% ======================
jobfile_F = {[dir_base fs dir_scripts fs 'second_ck_v2_job_Fcon.m']};
% Start with the 6 intended paired models
nrun_F = numel(targetNames_F);
% If mod_overall fallback used, add it as one more paired run
if use_mod_overall_as_paired
    nrun_F = nrun_F + 1;
end

jobs = repmat(jobfile_F, 1, nrun_F);
% Fcon template takes 3 inputs: Directory, Scans A, Scans B
inputs = cell(3, nrun_F);

wr = 1;
for i = 1:numel(targetNames_F)
    inputs{1, wr} = cellstr( f_dirs{i} );     % Directory
    inputs{2, wr} = cellstr( f_scans1{i} );   % Scans A
    inputs{3, wr} = cellstr( f_scans2{i} );   % Scans B
    wr = wr + 1;
end

% Append mod_overall as paired (average pin+pout) if needed
if use_mod_overall_as_paired
    dir_mod_overall = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_second_level fs 'mod_overall'];
    if ~exist(dir_mod_overall,'dir'), mkdir(dir_mod_overall); end
    inputs{1, wr} = cellstr( dir_mod_overall );
    inputs{2, wr} = cellstr( mod_overall_pair_scans1 );
    inputs{3, wr} = cellstr( mod_overall_pair_scans2 );
end

spm('defaults','fMRI');
spm_jobman('run', jobs, inputs{:});

disp('All 2nd-level models completed.');
