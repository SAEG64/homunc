function firstlevel_z_homunc_1_1_2
%% from: firstlevel_z_homeo_20_1_1_newprephy_new

%% Change according to your scan parameters and directory structure
cwd = 'DIRECTORY TO FMRI DATA: HOMUNC_parent';
cd(cwd);

fs                = filesep;
dir_base          = cwd;
dir_analysis_base = 'HU_FMRI_USE';

number_of_analysis_behaviour = '12';
number_of_analysis_scanner   = '1';
number_of_analysis_model     = '102';

dir_analysis_scanner = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model];

sub_original = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];
sub          = [301,302,304,305,307,308,310,311,312,313,316,317,320,321,322,323,324,325,326,327,328,334,335,306,315,319,333];

%% load params
load([dir_base '/fmri_2_BNW_weather_model_banded/HOMUNC_scripts_v2/beh2fmri_homunc_v8.mat']);

%% covariates: 6 motion + 2 outcome (on,value), plus constant
covis    = 8;   % 6 motion + 2 outcome
constant = 1;

%% base indices (1-indexed) for modeled regressors (before covis)
forest = 1;
N1 = 2;               % choice_n_W1: on, p_in, p_out, RT
N2 = N1 + 4;          % choice_n_W2
W1 = N2 + 4;          % choice_w_W1
W2 = W1 + 4;          % choice_w_W2
B1 = W2 + 4;          % choice_b_W1
B2 = B1 + 4;          % choice_b_W2

% within-choice block offsets
on   = 0;
pin  = 1;
pout = 2;
rt   = 3;

% number of modeled regressors BEFORE covis (forest + 6*4 = 25)
baseCols = B2 + 4;

spm fmri

for n = 1:length(sub)

    n_use   = find(sub_original==sub(n));
    blockNo = 1;

    sub_string = num2str(sub(n));
    dir_analysis_scanner_sub = ['stats_b_' number_of_analysis_behaviour '_s_' number_of_analysis_scanner '_m_' number_of_analysis_model '_s_' sub_string];

    % Names for identity contrasts (ONLY modeled regressors, no covis)
    T.contrasts = { ...
        'forest', ...
        'choice_n_W1_on','choice_n_W1_pin','choice_n_W1_pout','choice_n_W1_RT', ...
        'choice_n_W2_on','choice_n_W2_pin','choice_n_W2_pout','choice_n_W2_RT', ...
        'choice_w_W1_on','choice_w_W1_pin','choice_w_W1_pout','choice_w_W1_RT', ...
        'choice_w_W2_on','choice_w_W2_pin','choice_w_W2_pout','choice_w_W2_RT', ...
        'choice_b_W1_on','choice_b_W1_pin','choice_b_W1_pout','choice_b_W1_RT', ...
        'choice_b_W2_on','choice_b_W2_pin','choice_b_W2_pout','choice_b_W2_RT' ...
    };

    % Higher-level contrast names
    contrasts_FT = {'ons_overall','pin_overall','pout_overall','RT_overall', ...
        'ons_main_weather','pin_main_weather','pout_main_weather','RT_main_weather', ...
        'ons_main_ternary_N_gt_W','ons_main_ternary_W_gt_B', ...
        'pin_main_ternary_N_gt_W','pin_main_ternary_W_gt_B', ...
        'pout_main_ternary_N_gt_W','pout_main_ternary_W_gt_B', ...
        'RT_main_ternary_N_gt_W','RT_main_ternary_W_gt_B', ...
        'ons_interaction_1','ons_interaction_2', ...
        'pin_interaction_1','pin_interaction_2', ...
        'pout_interaction_1','pout_interaction_2', ...
        'RT_interaction_1','RT_interaction_2'};

    %% -------- Build contrast matrix --------
    % Identity for modeled regressors (padded with zeros for covis + constant)
    prep_id = [ eye(baseCols), zeros(baseCols, covis + constant) ];

    % Helper to make a contrast row over modeled regressors
    make = @(idxs) double(ismember(1:baseCols, idxs));

    C = []; % rows for FT contrasts over modeled regressors

    %% OVERALL
    C = [C; make([N1+on,  N2+on,  W1+on,  W2+on,  B1+on,  B2+on])];    % ons_overall
    C = [C; make([N1+pin, N2+pin, W1+pin, W2+pin, B1+pin, B2+pin])];  % pin_overall
    C = [C; make([N1+pout,N2+pout,W1+pout,W2+pout,B1+pout,B2+pout])]; % pout_overall
    C = [C; make([N1+rt,  N2+rt,  W1+rt,  W2+rt,  B1+rt,  B2+rt])];    % RT_overall

    %% WEATHER main effect (W1 > W2)
    C = [C; make([N1+on,  W1+on,  B1+on])  - make([N2+on,  W2+on,  B2+on])];
    C = [C; make([N1+pin, W1+pin, B1+pin]) - make([N2+pin, W2+pin, B2+pin])];
    C = [C; make([N1+pout,W1+pout,B1+pout])- make([N2+pout,W2+pout,B2+pout])];
    C = [C; make([N1+rt,  W1+rt,  B1+rt])  - make([N2+rt,  W2+rt,  B2+rt])];

    %% TERNARY main effects
    % N > W
    C = [C; make([N1+on,  N2+on])  - make([W1+on,  W2+on])];
    % W > B
    C = [C; make([W1+on,  W2+on])  - make([B1+on,  B2+on])];

    % N > W for pin / pout / RT
    C = [C; make([N1+pin, N2+pin]) - make([W1+pin, W2+pin])];
    C = [C; make([N1+pout,N2+pout])- make([W1+pout,W2+pout])];
    C = [C; make([N1+rt,  N2+rt])  - make([W1+rt,  W2+rt])];

    % W > B for pin / pout / RT
    C = [C; make([W1+pin, W2+pin]) - make([B1+pin, B2+pin])];
    C = [C; make([W1+pout,W2+pout])- make([B1+pout,B2+pout])];
    C = [C; make([W1+rt,  W2+rt])  - make([B1+rt,  B2+rt])];

    %% WEATHER × STATE interactions (two simple contrasts per regressor type)
    % Interaction #1: (N W1 vs W2) minus (W W1 vs W2)
    C = [C; (make([N1+on])  - make([N2+on]))  - (make([W1+on])  - make([W2+on]))];
    C = [C; (make([N1+pin]) - make([N2+pin])) - (make([W1+pin]) - make([W2+pin]))];
    C = [C; (make([N1+pout])- make([N2+pout]))- (make([W1+pout])- make([W2+pout]))];
    C = [C; (make([N1+rt])  - make([N2+rt]))  - (make([W1+rt])  - make([W2+rt]))];

    % Interaction #2: (W W1 vs W2) minus (B W1 vs W2)
    C = [C; (make([W1+on])  - make([W2+on]))  - (make([B1+on])  - make([B2+on]))];
    C = [C; (make([W1+pin]) - make([W2+pin])) - (make([B1+pin]) - make([B2+pin]))];
    C = [C; (make([W1+pout])- make([W2+pout]))- (make([B1+pout])- make([B2+pout]))];
    C = [C; (make([W1+rt])  - make([W2+rt]))  - (make([B1+rt])  - make([B2+rt]))];

    % Pad FT rows with zeros for covis + constant
    prep_ft = [ C, zeros(size(C,1), covis + constant) ];

    % Final contrast matrix: identity + FT
    prepare_con = [prep_id; prep_ft];

    % Names in the same order
    T.contrasts = [T.contrasts, contrasts_FT];

    % Replicate over sessions if needed
    T.contrastVectors = repmat(prepare_con, 1, blockNo);

    % SPM.mat path
    jobs{1}.stats{1}.con.spmmat = { [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_analysis_scanner_sub fs 'SPM.mat'] };

    % Write t-contrasts
    k = 1;
    for cont_nr = 1:length(T.contrasts)
        jobs{1}.stats{1}.con.consess{k}.tcon.name    = T.contrasts{cont_nr};
        jobs{1}.stats{1}.con.consess{k}.tcon.convec  = T.contrastVectors(cont_nr,:);
        jobs{1}.stats{1}.con.consess{k}.tcon.sessrep = 'none';
        k = k + 1;
    end

    % Run
    outputDir = [dir_base fs dir_analysis_base fs dir_analysis_scanner fs dir_analysis_scanner_sub];
    cd(outputDir);
    save contrasts.mat jobs
    disp(['RUNNING contrast specification for subject ' sub_string]);
    spm_jobman('run','contrasts.mat');
    disp(['Contrasts created for subject index ' num2str(n)]);
    clear jobs

end

cd(cwd);
