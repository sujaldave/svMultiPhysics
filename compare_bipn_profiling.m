% compare_bipn_profiling.m
%
% Compares svMultiPhysics bi-partition NS runs across backends
% (FSILS-CPU / Trilinos-CPU / Trilinos-GPU), preconditioner combinations,
% and (optionally) MPI process counts, using:
%   - the per-stage timing CSV written by svmp_profiling::write_csv()
%     (columns: backend,gmres_preconditioner,cg_preconditioner,stage,
%      calls,total_seconds,avg_seconds)
%   - the solver's histor.dat convergence log
%
% Directory layout assumed (matches allrun.sh's "mv <N>-procs <label>"
% convention): each case lives in its own folder named
%   <nProcs>-procs-<variantSuffix>/histor.dat
% e.g. 1-procs-cpu-resis-ml/histor.dat, 2-procs-gpu-ml-ml/histor.dat, ...
%
% Profiling CSV resolution, per case, tried in this order:
%   1) <nProcs>-procs-<variantSuffix>.csv  (a per-case file -- see the
%      SVMP_PROFILE_CSV note below; this is the robust option and the
%      only one that can tell two runs of the same backend+preconditioner
%      combo apart at different process counts)
%   2) the single shared SHARED_CSV (e.g. svmp_profiling.csv) filtered by
%      that variant's backend/gmres_preconditioner/cg_preconditioner --
%      this is what you have today for the 1-proc runs, but it CANNOT
%      distinguish e.g. "cpu-ml-ml at 1 proc" from "cpu-ml-ml at 4 procs"
%      since neither the backend nor the preconditioner names encode
%      process count. A warning is printed whenever this fallback is used.
%
% To make (1) the normal case going forward (recommended once you start
% comparing process counts), add one line to allrun.sh before each run:
%   export SVMP_PROFILE_CSV=1-procs-cpu-resis-ml.csv
%   mpirun -n 1 ./svMultiPhysics solver_bipnTrilinos_resis_ml.xml
% i.e. name it after the folder the run's "mv" step is about to create,
% but without nesting it inside that (not-yet-existing) folder.
%
% To add another case (another preconditioner combination, another
% backend, another process count): add one row to VARIANTS and/or one
% entry to PROC_COUNTS below and rerun this script -- nothing else needs
% to change. Stage colors, case colors, and legends are all derived from
% the data.

clear; clc; close all;

%% ---------------------------------------------------------------------
%  CONFIGURATION -- edit this section for your directory layout.
%  ---------------------------------------------------------------------

% Process counts to compare. With one entry, case labels are just the
% variant label; with more than one, "(Np)" is appended so lines/bars
% stay distinguishable.
PROC_COUNTS = [1];
% PROC_COUNTS = [1 2 4 8 16];

% Fallback shared CSV (today's single top-level file). Only used for a
% case whose per-case "<dir>.csv" file does not exist.
SHARED_CSV = 'svmp_profiling.csv';

% One row per case variant: {folderSuffix, label, backend, gmresPrec, cgPrec}
% backend/gmresPrec/cgPrec are only used for the SHARED_CSV fallback
% filter; leave them accurate even if you expect to always have per-case
% CSVs, so the fallback still works if a per-case file is ever missing.
%   backend strings actually written by the solver:
%     FSILS build            -> "FSILS-CPU"
%     Trilinos CPU build     -> "Trilinos-Serial" or "Trilinos-OpenMP"
%                                (whichever Kokkos host space your build uses)
%     Trilinos GPU build     -> "Trilinos-Cuda"
%   preconditioner strings: "trilinos-diagonal", "trilinos-ilu",
%     "trilinos-ml", "trilinos-resistance", "fsils"
VARIANTS = {
    'cpu-fsils',      'FSILS-CPU',          'FSILS-CPU',       '',                     '';
    'cpu-resis-ml',   'Tril-CPU Resis/ML',  'Trilinos-Serial', 'trilinos-resistance',  'trilinos-ml';
    'cpu-resis-diag', 'Tril-CPU Resis/Diag','Trilinos-Serial', 'trilinos-resistance',  'trilinos-diagonal';
    'cpu-diag-ml',    'Tril-CPU Diag/ML',   'Trilinos-Serial', 'trilinos-diagonal',    'trilinos-ml';
    'cpu-ml-ml',      'Tril-CPU ML/ML',     'Trilinos-Serial', 'trilinos-ml',          'trilinos-ml';
    'gpu-resis-ml',   'Tril-GPU Resis/ML',  'Trilinos-Cuda',   'trilinos-resistance',  'trilinos-ml';
    'gpu-resis-diag', 'Tril-GPU Resis/Diag','Trilinos-Cuda',   'trilinos-resistance',  'trilinos-diagonal';
    'gpu-diag-ml',    'Tril-GPU Diag/ML',   'Trilinos-Cuda',   'trilinos-diagonal',    'trilinos-ml';
    'gpu-ml-ml',      'Tril-GPU ML/ML',     'Trilinos-Cuda',   'trilinos-ml',          'trilinos-ml';
};

cases = build_cases(PROC_COUNTS, VARIANTS);
nCases = numel(cases);
caseLabels = {cases.label};

%% ---------------------------------------------------------------------
%  Load data
%  ---------------------------------------------------------------------
profilingTables = cell(nCases, 1);
histTables = cell(nCases, 1);
allStages = {};

for i = 1:nCases
    profilingTables{i} = resolve_and_read_profiling(cases(i), SHARED_CSV);
    allStages = union(allStages, cellstr(profilingTables{i}.stage), 'stable');

    histTables{i} = read_histor(cases(i).historDat);
end

stageOrder = order_stages(allStages);
stageColors = get_stage_colors(stageOrder);
caseColors = turbo(max(nCases, 2));

%% ---------------------------------------------------------------------
%  Figure 1: total profiling time per stage, per case (stacked bar)
%  ---------------------------------------------------------------------
timeMatrix = zeros(nCases, numel(stageOrder));
for i = 1:nCases
    timeMatrix(i, :) = stage_times(profilingTables{i}, stageOrder);
end

figure('Name', 'BIPN profiling: total time by stage', 'Color', 'w');
b = bar(timeMatrix, 'stacked');
apply_stage_colors(b, stageOrder, stageColors);
set(gca, 'XTick', 1:nCases, 'XTickLabel', caseLabels, 'XTickLabelRotation', 30);
ylabel('Total time (s)');
title('Total profiling time by stage');
legend(stageOrder, 'Location', 'eastoutside', 'Interpreter', 'none');
grid on;

%% ---------------------------------------------------------------------
%  Figure 2: percentage of profiled time per stage, per case
%  ---------------------------------------------------------------------
pctMatrix = 100 * timeMatrix ./ sum(timeMatrix, 2);

figure('Name', 'BIPN profiling: % time by stage', 'Color', 'w');
b = bar(pctMatrix, 'stacked');
apply_stage_colors(b, stageOrder, stageColors);
set(gca, 'XTick', 1:nCases, 'XTickLabel', caseLabels, 'XTickLabelRotation', 30);
ylabel('% of profiled time');
ylim([0 100]);
title('Percentage of profiled time by stage');
legend(stageOrder, 'Location', 'eastoutside', 'Interpreter', 'none');
grid on;

%% ---------------------------------------------------------------------
%  Figure 3: total wall time across cases (from histor.dat)
%  ---------------------------------------------------------------------
wallTime = zeros(nCases, 1);
for i = 1:nCases
    wallTime(i) = histTables{i}.T(end);
end

figure('Name', 'Total wall time', 'Color', 'w');
b = bar(wallTime, 'FaceColor', 'flat');
for i = 1:nCases
    b.CData(i, :) = caseColors(i, :);
end
set(gca, 'XTick', 1:nCases, 'XTickLabel', caseLabels, 'XTickLabelRotation', 30);
ylabel('Total wall time (s)');
title('Total wall time across cases');
grid on;

%% ---------------------------------------------------------------------
%  Figures 4-6: residual evolution over time steps (one figure each)
%  ---------------------------------------------------------------------
residualMetrics = {'RiR1', 'RiR0', 'RRi'};
residualLabels = {'Ri/R1', 'Ri/R0', 'R/Ri'};

for m = 1:numel(residualMetrics)
    figure('Name', sprintf('Residual evolution: %s', residualLabels{m}), 'Color', 'w');
    hold on;
    for i = 1:nCases
        d = last_iteration_per_timestep(histTables{i});
        plot(d.timestep, d.(residualMetrics{m}), '-o', ...
            'Color', caseColors(i, :), 'MarkerFaceColor', caseColors(i, :), ...
            'DisplayName', caseLabels{i});
    end
    hold off;
    set(gca, 'YScale', 'log');
    xlabel('Time step');
    ylabel(residualLabels{m});
    title(sprintf('%s evolution over time', residualLabels{m}));
    legend('Location', 'best', 'Interpreter', 'none');
    grid on;
end

%% ---------------------------------------------------------------------
%  Figure 7: linear solver iteration count evolution
%  ---------------------------------------------------------------------
figure('Name', 'Linear iteration count evolution', 'Color', 'w');
hold on;
for i = 1:nCases
    d = last_iteration_per_timestep(histTables{i});
    plot(d.timestep, d.lsIt, '-o', ...
        'Color', caseColors(i, :), 'MarkerFaceColor', caseColors(i, :), ...
        'DisplayName', caseLabels{i});
end
hold off;
xlabel('Time step');
ylabel('Linear solver iterations (lsIt)');
title('Linear solver iteration count evolution');
legend('Location', 'best', 'Interpreter', 'none');
grid on;

%% ---------------------------------------------------------------------
%  Figure 8: % time spent inside the linear solver, evolution
%  ---------------------------------------------------------------------
figure('Name', '%t evolution', 'Color', 'w');
hold on;
for i = 1:nCases
    d = last_iteration_per_timestep(histTables{i});
    plot(d.timestep, d.pct_t, '-o', ...
        'Color', caseColors(i, :), 'MarkerFaceColor', caseColors(i, :), ...
        'DisplayName', caseLabels{i});
end
hold off;
xlabel('Time step');
ylabel('% time in linear solver (%t)');
title('Percentage of time spent inside the linear solver, evolution');
legend('Location', 'best', 'Interpreter', 'none');
grid on;

%% ---------------------------------------------------------------------
%  Figure 9: average linear iteration count per case
%  ---------------------------------------------------------------------
avgLsIt = zeros(nCases, 1);
for i = 1:nCases
    avgLsIt(i) = mean(histTables{i}.lsIt);
end

figure('Name', 'Average linear iterations', 'Color', 'w');
b = bar(avgLsIt, 'FaceColor', 'flat');
for i = 1:nCases
    b.CData(i, :) = caseColors(i, :);
end
set(gca, 'XTick', 1:nCases, 'XTickLabel', caseLabels, 'XTickLabelRotation', 30);
ylabel('Average lsIt (per nonlinear solve)');
title('Average number of linear solver iterations');
grid on;

%% ---------------------------------------------------------------------
%  Figure 10: average % time in linear solver per case
%  ---------------------------------------------------------------------
avgPctT = zeros(nCases, 1);
for i = 1:nCases
    avgPctT(i) = mean(histTables{i}.pct_t);
end

figure('Name', 'Average %t', 'Color', 'w');
b = bar(avgPctT, 'FaceColor', 'flat');
for i = 1:nCases
    b.CData(i, :) = caseColors(i, :);
end
set(gca, 'XTick', 1:nCases, 'XTickLabel', caseLabels, 'XTickLabelRotation', 30);
ylabel('Average % time in linear solver');
title('Average percentage of time spent inside the linear solver');
grid on;

%% =======================================================================
%  Local functions
%  =======================================================================

function cases = build_cases(procCounts, variants)
% BUILD_CASES  Cross procCounts x variants into one CASES struct array.
% variants is an Nx5 cell array: {folderSuffix, label, backend, gmresPrec, cgPrec}.
% Folder for a case is "<nProcs>-procs-<folderSuffix>"; its histor.dat is
% expected directly inside that folder, and its per-case profiling CSV
% (if present) is "<nProcs>-procs-<folderSuffix>.csv" next to it.
    cases = struct('label', {}, 'dir', {}, 'historDat', {}, ...
        'perCaseCsv', {}, 'backend', {}, 'gmresPrec', {}, 'cgPrec', {});
    showProcCount = numel(procCounts) > 1;

    for p = 1:numel(procCounts)
        nProcs = procCounts(p);
        for v = 1:size(variants, 1)
            suffix = variants{v, 1};
            baseLabel = variants{v, 2};
            backend = variants{v, 3};
            gmresPrec = variants{v, 4};
            cgPrec = variants{v, 5};

            dirName = sprintf('%d-procs-%s', nProcs, suffix);
            if showProcCount
                label = sprintf('%s (%dp)', baseLabel, nProcs);
            else
                label = baseLabel;
            end

            cases(end+1) = struct( ...
                'label', label, ...
                'dir', dirName, ...
                'historDat', fullfile(dirName, 'histor.dat'), ...
                'perCaseCsv', [dirName '.csv'], ...
                'backend', string(backend), ...
                'gmresPrec', string(gmresPrec), ...
                'cgPrec', string(cgPrec)); %#ok<AGROW>
        end
    end
end

function T = resolve_and_read_profiling(c, sharedCsv)
% RESOLVE_AND_READ_PROFILING  Prefer a per-case profiling CSV; fall back
% to filtering the shared CSV by backend/gmres/cg preconditioner. See the
% header comment for why the fallback cannot separate process counts.
    if isfile(c.perCaseCsv)
        T = read_profiling_csv(c.perCaseCsv, "", "", "");
    else
        warning('compare_bipn_profiling:sharedCsvFallback', ...
            ['%s: no per-case profiling CSV at "%s" -- falling back to "%s" ' ...
             'filtered by backend/preconditioner. This cannot tell apart two ' ...
             'runs of the same backend+preconditioner combination at ' ...
             'different process counts; see the script header for the fix.'], ...
            c.label, c.perCaseCsv, sharedCsv);
        T = read_profiling_csv(sharedCsv, c.backend, c.gmresPrec, c.cgPrec);
    end
end

function T = read_profiling_csv(path, backend, gmresPrec, cgPrec)
% READ_PROFILING_CSV  Load one svmp_profiling.csv, optionally filtered to
% one backend/gmres_preconditioner/cg_preconditioner combination, and
% collapse duplicate stage rows (e.g. from an appended/restarted run) by
% summing their time and call counts.
    if ~isfile(path)
        error('compare_bipn_profiling:missingFile', ...
            'Profiling CSV not found: %s', path);
    end
    raw = readtable(path, 'TextType', 'string', 'Delimiter', ',');

    if strlength(backend) > 0
        raw = raw(raw.backend == backend, :);
    end
    if strlength(gmresPrec) > 0
        raw = raw(raw.gmres_preconditioner == gmresPrec, :);
    end
    if strlength(cgPrec) > 0
        raw = raw(raw.cg_preconditioner == cgPrec, :);
    end
    if isempty(raw)
        error('compare_bipn_profiling:noRows', ...
            'No matching rows in %s (backend="%s", gmres="%s", cg="%s")', ...
            path, backend, gmresPrec, cgPrec);
    end

    [uniqueStages, ~, idx] = unique(raw.stage);
    totalSeconds = accumarray(idx, raw.total_seconds);
    calls = accumarray(idx, raw.calls);
    T = table(uniqueStages, totalSeconds, calls, ...
        'VariableNames', {'stage', 'total_seconds', 'calls'});
end

function stageOrder = order_stages(allStages)
% ORDER_STAGES  Canonical pipeline order for the 10 stages profiled by
% svmp_profiling, with any additional/unrecognized stage names appended
% at the end (so new stages added later still show up automatically).
    canonical = {'Element Assembly', 'System Setup', 'Block Extraction', ...
        'Boundary Condition', 'Preconditioner Setup', 'MueLu Setup', ...
        'Predictor', 'Linear Solve', 'Tpetra Allocation', ...
        'Host & Device Synchronization'};
    known = intersect(canonical, allStages, 'stable');
    extra = setdiff(allStages, canonical, 'stable');
    stageOrder = [known, extra];
end

function stageColors = get_stage_colors(stageOrder)
% GET_STAGE_COLORS  One fixed color per stage name (containers.Map keyed
% by stage name), stable across figures regardless of which subset of
% stages a given case actually reports.
    cmap = turbo(numel(stageOrder));
    stageColors = containers.Map(stageOrder, num2cell(cmap, 2));
end

function apply_stage_colors(b, stageOrder, stageColors)
% APPLY_STAGE_COLORS  Color each series of a stacked bar plot by stage.
    for s = 1:numel(stageOrder)
        b(s).FaceColor = stageColors(stageOrder{s});
    end
end

function times = stage_times(profilingTable, stageOrder)
% STAGE_TIMES  Total_seconds for each stage in stageOrder (0 if a case
% did not report that stage, e.g. "MueLu Setup" for a non-ML run).
    times = zeros(1, numel(stageOrder));
    for s = 1:numel(stageOrder)
        idx = find(strcmp(profilingTable.stage, stageOrder{s}), 1);
        if ~isempty(idx)
            times(s) = profilingTable.total_seconds(idx);
        end
    end
end

function data = read_histor(path)
% READ_HISTOR  Parse an svMultiPhysics histor.dat NS log into a table
% with one row per printed nonlinear-iteration line:
%   timestep, iter, T, dB_RI, RiR1, RiR0, RRi, lsIt, dB_LS, pct_t
% Handles both converged ([...]) and non-converged (!...!) bracket styles.
    if ~isfile(path)
        error('compare_bipn_profiling:missingFile', ...
            'histor.dat not found: %s', path);
    end
    txt = fileread(path);
    pattern = ['\S+\s+(\d+)-(\d+)\s+([0-9.eE+-]+)\s+[\[!]\s*([-0-9]+)\s+' ...
        '([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)[\]!]\s+' ...
        '[\[!]\s*(\d+)\s+([-0-9]+)\s+(\d+)[\]!]'];
    tok = regexp(txt, pattern, 'tokens');
    n = numel(tok);
    if n == 0
        error('compare_bipn_profiling:emptyHistor', ...
            'No NS iteration lines could be parsed from %s', path);
    end

    timestep = zeros(n, 1); iter = zeros(n, 1); T = zeros(n, 1);
    dB_RI = zeros(n, 1); RiR1 = zeros(n, 1); RiR0 = zeros(n, 1);
    RRi = zeros(n, 1); lsIt = zeros(n, 1); dB_LS = zeros(n, 1);
    pct_t = zeros(n, 1);

    for k = 1:n
        c = tok{k};
        timestep(k) = str2double(c{1});
        iter(k)     = str2double(c{2});
        T(k)        = str2double(c{3});
        dB_RI(k)    = str2double(c{4});
        RiR1(k)     = str2double(c{5});
        RiR0(k)     = str2double(c{6});
        RRi(k)      = str2double(c{7});
        lsIt(k)     = str2double(c{8});
        dB_LS(k)    = str2double(c{9});
        pct_t(k)    = str2double(c{10});
    end

    data = table(timestep, iter, T, dB_RI, RiR1, RiR0, RRi, lsIt, dB_LS, pct_t);
end

function lastIterData = last_iteration_per_timestep(data)
% LAST_ITERATION_PER_TIMESTEP  Reduce a histor table to one row per time
% step: the row with the highest nonlinear-iteration number, i.e. the
% final/converged state reported for that step. This is what the
% evolution-over-time plots compare across cases regardless of how many
% nonlinear iterations each case happened to take per step.
    uniqueSteps = unique(data.timestep);
    idx = zeros(numel(uniqueSteps), 1);
    for k = 1:numel(uniqueSteps)
        rows = find(data.timestep == uniqueSteps(k));
        [~, mi] = max(data.iter(rows));
        idx(k) = rows(mi);
    end
    lastIterData = data(idx, :);
end
