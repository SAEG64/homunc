% from: hstasis_nonortho_v2_job_trans_23_1_2_newpre  (Banded p(success) edition)
%% new getonsets with ortho option
clear all
cwd = 'DIRECTORY TO FMRI DATA: HOMUNC_parent';
addpath('/home/LOCAL_NAME_HERE/spm12')

%% Change according to your directory structure and scan parameters
fs                    = filesep;
dir_parent            = cwd;
dir_analysis_base     = 'HU_FMRI_USE';
dir_fMRI_base         = 'HU_FMRI_USE';
dir_scripts           = 'HOMUNC_scripts_v2';

number_of_analysis_behaviour = '12';
number_of_analysis_scanner   = '1';
number_of_analysis_model     = '102';

ses_string_vec = {'1','2','3','4','5','6','7','8','9','10'};
dir_analysis_behviour = ['beh_v' number_of_analysis_behaviour '_agg'];
dir_analysis_scanner  = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model];
cd([dir_parent fs dir_analysis_base]);
mkdir(dir_analysis_scanner);

%% load overall relevant parameters
load('DIRECTORY TO FMRI DATA: HOMUNC_parent/fmri_2_BNW_weather_model_banded/HOMUNC_scripts_v2/HOMUNC_data_beh_B_fMRI_v1.mat');
ext_header = [{'x57_weather_1_gain_magnitude'}, {'x58_weather_2_gain_magnitude'}, {'x59_weather_1_p_gain'}, {'x60_weather_2_p_gain'}, {'weather_dynamic'}, {'p_delta'}, {'p_delta_dynamic'}, {'BNW_conditions'}, {'BNW_condition'}];
header_columns = [header_columns, ext_header];

%% subjects
sub_original = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];
sub          = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];

estimate = 1;
ortho_yn = 0;  % IMPORTANT: turn OFF orthogonalization

%% spm
spm fmri

for n = 1:length(sub)

    sub_string = num2str(sub(n));
    n_use      = find(sub_original==sub(n));
    blockNo    = size(runs{ n_use }, 2); % sessions

    filename_sub = ['beh_v' number_of_analysis_behaviour '_sub_' num2str(sub(n)) ];
    clear Z
    load(['DIRECTORY TO FMRI DATA: HOMUNC_parent/fmri_2_BNW_weather_model_banded/HOMUNC_scripts_V2/beh_prep/' fs filename_sub], 'Z');

    cd(['DIRECTORY TO FMRI DATA: HOMUNC_parent' fs dir_analysis_base fs dir_analysis_scanner]);
    dir_analysis_scanner_sub = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' sub_string];
    mkdir(dir_analysis_scanner_sub);
    outputDir = [dir_parent fs dir_analysis_base fs dir_analysis_scanner fs dir_analysis_scanner_sub];

    %-----------------------------------------------------------------------
    % Batch spec
    %-----------------------------------------------------------------------
    matlabbatch{1}.spm.stats.fmri_spec.dir              = {outputDir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units     = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT        = 2.1;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t    = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0   = 1;

    % Accumulators across sessions (concatenated design)
    R = [];
    scans     = 0;
    scan_n    = [];
    allscans  = [];
    ons_forest = [];

    % Onsets per condition (state × weather)
    ons_choice_B_W1 = []; ons_choice_B_W2 = [];
    ons_choice_N_W1 = []; ons_choice_N_W2 = [];
    ons_choice_W_W1 = []; ons_choice_W_W2 = [];

    % pMods: banded p(success)
    pin_B_W1 = [];  pout_B_W1 = [];
    pin_B_W2 = [];  pout_B_W2 = [];
    pin_N_W1 = [];  pout_N_W1 = [];
    pin_N_W2 = [];  pout_N_W2 = [];
    pin_W_W1 = [];  pout_W_W1 = [];
    pin_W_W2 = [];  pout_W_W2 = [];

    % RT pMods
    RT_B_W1 = []; RT_B_W2 = [];
    RT_N_W1 = []; RT_N_W2 = [];
    RT_W_W1 = []; RT_W_W2 = [];

    % Outcome EV + pMod
    ons_outcome = [];
    outcome     = [];

    for k = 1:blockNo

        %% directories / scans
        dir_fMRI_sub       = ['HU_sub_' num2str(sub(n)) ];
        k_use              = runs{ n_use }(k);
        dir_sess_specified = ['session_' num2str(k_use)];
        epiDir             = [dir_parent fs dir_fMRI_base fs dir_fMRI_sub fs 'functional_1' fs dir_sess_specified];

        f      = spm_select('List', epiDir, '^swuamr.*\.img$');
        files  = cellstr([repmat([epiDir fs],size(f,1),1) f]);
        allscans = [allscans; files];

        %% Regressors: forest onset (block)
        ons_forest = [ons_forest, Z.forest{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];

        %% Decision onsets (state × weather)
        ons_choice_B_W1 = [ons_choice_B_W1, Z.choice_B_W1{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        ons_choice_B_W2 = [ons_choice_B_W2, Z.choice_B_W2{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];

        ons_choice_N_W1 = [ons_choice_N_W1, Z.choice_N_W1{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        ons_choice_N_W2 = [ons_choice_N_W2, Z.choice_N_W2{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];

        ons_choice_W_W1 = [ons_choice_W_W1, Z.choice_W_W1{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        ons_choice_W_W2 = [ons_choice_W_W2, Z.choice_W_W2{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];

        %% RT pMods (log RT as you had)
        RT_B_W1 = [RT_B_W1, log(Z.choice_B_W1{k_use}(:,10)')];
        RT_B_W2 = [RT_B_W2, log(Z.choice_B_W2{k_use}(:,10)')];

        RT_N_W1 = [RT_N_W1, log(Z.choice_N_W1{k_use}(:,10)')];
        RT_N_W2 = [RT_N_W2, log(Z.choice_N_W2{k_use}(:,10)')];

        RT_W_W1 = [RT_W_W1, log(Z.choice_W_W1{k_use}(:,10)')];
        RT_W_W2 = [RT_W_W2, log(Z.choice_W_W2{k_use}(:,10)')];

        %% Map banded p(success) (full-length vectors) to each condition by tolerant onset matching
        base_on   = Z.choice{k_use}(:,1);        % all choice onsets in this session
        pin_all   = Z.p_in{k_use}(:);            % full-length p_in (one per trial)
        pout_all  = Z.p_out{k_use}(:);           % full-length p_out (one per trial)
        tol       = 1e-6;                        % seconds tolerance for matching

        % helper: pick p_in/p_out entries whose base_on matches a given onset list
        [pin_tmp, pout_tmp] = pick_params(base_on, pin_all, pout_all, Z.choice_B_W1{k_use}(:,1), tol);
        pin_B_W1  = [pin_B_W1,  pin_tmp'];
        pout_B_W1 = [pout_B_W1, pout_tmp'];

        [pin_tmp, pout_tmp] = pick_params(base_on, pin_all, pout_all, Z.choice_B_W2{k_use}(:,1), tol);
        pin_B_W2  = [pin_B_W2,  pin_tmp'];
        pout_B_W2 = [pout_B_W2, pout_tmp'];

        [pin_tmp, pout_tmp] = pick_params(base_on, pin_all, pout_all, Z.choice_N_W1{k_use}(:,1), tol);
        pin_N_W1  = [pin_N_W1,  pin_tmp'];
        pout_N_W1 = [pout_N_W1, pout_tmp'];

        [pin_tmp, pout_tmp] = pick_params(base_on, pin_all, pout_all, Z.choice_N_W2{k_use}(:,1), tol);
        pin_N_W2  = [pin_N_W2,  pin_tmp'];
        pout_N_W2 = [pout_N_W2, pout_tmp'];

        [pin_tmp, pout_tmp] = pick_params(base_on, pin_all, pout_all, Z.choice_W_W1{k_use}(:,1), tol);
        pin_W_W1  = [pin_W_W1,  pin_tmp'];
        pout_W_W1 = [pout_W_W1, pout_tmp'];

        [pin_tmp, pout_tmp] = pick_params(base_on, pin_all, pout_all, Z.choice_W_W2{k_use}(:,1), tol);
        pin_W_W2  = [pin_W_W2,  pin_tmp'];
        pout_W_W2 = [pout_W_W2, pout_tmp'];

        %% Outcome
        ons_outcome = [ons_outcome, Z.outcome{k_use}(:,1)' + sum(scans) * matlabbatch{1}.spm.stats.fmri_spec.timing.RT];
        outcome     = [outcome,    Z.outcome{k_use}(:,2)'];

        %% motion regressors
        extension  = 'txt';
        motion_file = dir([epiDir fs '*.' extension]);
        motio       = textread([epiDir fs motion_file.name]); %#ok<TEXTRD>
        R = [R; motio];

        %% scans per session
        i_scans = length(files);
        scans   = [scans; i_scans];
        scan_n  = [scan_n, length(files)];

        %% reset
        f = []; files = [];

    end % sessions loop

    %% Sanity checks (optional)
    assert(numel(ons_choice_N_W1)==numel(pin_N_W1) && numel(pin_N_W1)==numel(pout_N_W1) && numel(pout_N_W1)==numel(RT_N_W1), 'N_W1 length mismatch');
    assert(numel(ons_choice_N_W2)==numel(pin_N_W2) && numel(pin_N_W2)==numel(pout_N_W2) && numel(pout_N_W2)==numel(RT_N_W2), 'N_W2 length mismatch');
    assert(numel(ons_choice_W_W1)==numel(pin_W_W1) && numel(pin_W_W1)==numel(pout_W_W1) && numel(pout_W_W1)==numel(RT_W_W1), 'W_W1 length mismatch');
    assert(numel(ons_choice_W_W2)==numel(pin_W_W2) && numel(pin_W_W2)==numel(pout_W_W2) && numel(pout_W_W2)==numel(RT_W_W2), 'W_W2 length mismatch');
    assert(numel(ons_choice_B_W1)==numel(pin_B_W1) && numel(pin_B_W1)==numel(pout_B_W1) && numel(pout_B_W1)==numel(RT_B_W1), 'B_W1 length mismatch');
    assert(numel(ons_choice_B_W2)==numel(pin_B_W2) && numel(pin_B_W2)==numel(pout_B_W2) && numel(pout_B_W2)==numel(RT_B_W2), 'B_W2 length mismatch');

    %% Conditions (concatenated design)
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans = allscans;

    % forest (block)
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).name     = 'forest_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).onset    = ons_forest;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).tmod     = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(1).orth     = ortho_yn;

    % normal state, W1
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).name     = 'choice_n_W1_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).onset    = ons_choice_N_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).tmod     = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).name  = 'p_in';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).param = pin_N_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(1).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).name  = 'p_out';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).param = pout_N_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(2).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(3).name  = 'RT';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(3).param = RT_N_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).pmod(3).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(2).orth     = ortho_yn;

    % normal state, W2
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).name     = 'choice_n_W2_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).onset    = ons_choice_N_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).tmod     = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).name  = 'p_in';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).param = pin_N_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(1).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(2).name  = 'p_out';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(2).param = pout_N_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(2).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(3).name  = 'RT';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(3).param = RT_N_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).pmod(3).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(3).orth     = ortho_yn;

    % wait, W1
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).name     = 'choice_w_W1_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).onset    = ons_choice_W_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).tmod     = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(1).name  = 'p_in';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(1).param = pin_W_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(1).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(2).name  = 'p_out';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(2).param = pout_W_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(2).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(3).name  = 'RT';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(3).param = RT_W_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).pmod(3).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(4).orth     = ortho_yn;

    % wait, W2
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).name     = 'choice_w_W2_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).onset    = ons_choice_W_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).tmod     = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(1).name  = 'p_in';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(1).param = pin_W_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(1).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(2).name  = 'p_out';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(2).param = pout_W_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(2).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(3).name  = 'RT';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(3).param = RT_W_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).pmod(3).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(5).orth     = ortho_yn;

    % binary energy state, W1
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).name     = 'choice_b_W1_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).onset    = ons_choice_B_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).tmod     = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(1).name  = 'p_in';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(1).param = pin_B_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(1).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(2).name  = 'p_out';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(2).param = pout_B_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(2).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(3).name  = 'RT';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(3).param = RT_B_W1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).pmod(3).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(6).orth     = ortho_yn;

    % binary energy state, W2
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).name     = 'choice_b_W2_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).onset    = ons_choice_B_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).tmod     = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(1).name  = 'p_in';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(1).param = pin_B_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(1).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(2).name  = 'p_out';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(2).param = pout_B_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(2).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(3).name  = 'RT';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(3).param = RT_B_W2;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).pmod(3).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(7).orth     = ortho_yn;

    % outcome
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(8).name     = 'outcome_1';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(8).onset    = ons_outcome;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(8).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(8).tmod     = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(8).pmod(1).name  = 'outcome_value';
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(8).pmod(1).param = outcome;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(8).pmod(1).poly  = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(8).orth     = ortho_yn;

    % no extra conditions
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi   = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).regress = struct('name', {}, 'val', {});

    %% multi-reg (motion)
    multiregFile = sprintf('regs%d_all.mat',sub(n));
    multiregPath = [outputDir fs multiregFile];
    save(multiregPath, 'R');
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg = {multiregPath};

    %% HPF, basis, etc.
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 128;

    matlabbatch{1}.spm.stats.fmri_spec.fact                = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs    = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt                = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global              = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh             = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask                = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi                 = 'AR(1)';

    %% run batch
    cfg_util('initjob', matlabbatch);
    cfg_util('run', matlabbatch);
    cfg_util('deljob', matlabbatch);

    %% concatenate
    spm_fmri_concatenate(fullfile(outputDir, 'SPM.mat'), scan_n);

    if estimate
        clear matlabbatch
        outputDir_SPM = [outputDir fs 'SPM.mat'];
        matlabbatch{1}.spm.stats.fmri_est.spmmat = {outputDir_SPM};
        matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;

        cfg_util('initjob', matlabbatch);
        cfg_util('run', matlabbatch);
        cfg_util('deljob', matlabbatch);
        clear matlabbatch
    end

end % subject

%% -------- Local helper (tolerant onset matching) ------------------------
function [p_in_sel, p_out_sel] = pick_params(base_on, p_in_all, p_out_all, subset_onsets, tol)
% Return the p_in / p_out values for the rows in `base_on` that correspond
% to `subset_onsets`, using a tolerant match on onset times.
%
% base_on:        [T x 1] all choice onsets in session k
% p_in_all:       [T x 1] p_in for each trial in session k
% p_out_all:      [T x 1] p_out for each trial in session k
% subset_onsets:  [t x 1] onset times for a given condition (e.g., N_W1)
% tol:            scalar tolerance in seconds (e.g., 1e-6)

    if isempty(subset_onsets)
        p_in_sel  = [];
        p_out_sel = [];
        return
    end

    % Use ismembertol to be robust to fp rounding of onsets after concatenation
    idx = false(size(base_on));
    for ii = 1:numel(subset_onsets)
        hit = abs(base_on - subset_onsets(ii)) <= tol;
        % If multiple hits (shouldn't happen), take the first unmatched
        if any(hit)
            % ensure one-to-one by picking the closest unused
            [~, order] = sort(abs(base_on - subset_onsets(ii)));
            for jj = 1:numel(order)
                if ~idx(order(jj)) && abs(base_on(order(jj)) - subset_onsets(ii)) <= tol
                    idx(order(jj)) = true;
                    break
                end
            end
        end
    end

    p_in_sel  = p_in_all(idx);
    p_out_sel = p_out_all(idx);
end
