function manifestPath = autoclean_eeglab_fooof(inputFile, outputDir, vhtpDir, eeglabDir, fmin, fmax, saveFooofImg, useParallel)
%AUTOCLEAN_EEGLAB_FOOOF Run eeg_htpCalcFooof on one EEGLAB .set file.
%
% This is the production wrapper for the AutoClean MATLAB FOOOF block. It avoids
% machine-specific paths and writes a small JSON manifest for Python to ingest.

arguments
    inputFile (1,:) char
    outputDir (1,:) char
    vhtpDir (1,:) char
    eeglabDir (1,:) char
    fmin (1,1) double = 1
    fmax (1,1) double = 55
    saveFooofImg (1,1) logical = false
    useParallel (1,1) logical = false
end

if exist(inputFile, 'file') ~= 2
    error('Input file not found: %s', inputFile);
end
if exist(vhtpDir, 'dir') ~= 7
    error('vhtp path not found: %s', vhtpDir);
end
if exist(eeglabDir, 'dir') ~= 7
    error('EEGLAB path not found: %s', eeglabDir);
end

if exist(outputDir, 'dir') ~= 7
    mkdir(outputDir);
end

addpath(genpath(eeglabDir));
eeglab nogui;
addpath(genpath(vhtpDir));

[inputDir, inputName, ~] = fileparts(inputFile);

EEG = pop_loadset('filename', [inputName '.set'], 'filepath', inputDir);
[EEG, opts] = eeg_htpCalcFooof(EEG, ...
    'spect_freqs', [fmin fmax], ...
    'save_to_csv', true, ...
    'save_fooof_img', saveFooofImg, ...
    'parallel', useParallel, ...
    'outputdir', outputDir);

summary = opts.FOOOF_results.summary_table;
aperiodicRows = summary(contains(string(summary.key), 'aperiodic'), :);

summaryCsv = fullfile(outputDir, sprintf('%s_fooof_summary.csv', inputName));
aperiodicCsv = fullfile(outputDir, sprintf('%s_fooof_aperiodic.csv', inputName));
writetable(summary, summaryCsv);
writetable(aperiodicRows, aperiodicCsv);

matlabOutputDir = fullfile(outputDir, 'eeg_htpCalcFooof');
fitCsv = '';
matlabSummaryCsv = '';
if exist(matlabOutputDir, 'dir') == 7
    fitFiles = dir(fullfile(matlabOutputDir, '*_fooof_fit.csv'));
    summaryFiles = dir(fullfile(matlabOutputDir, '*_fooof_summary.csv'));
    if ~isempty(fitFiles)
        fitCsv = fullfile(fitFiles(1).folder, fitFiles(1).name);
    end
    if ~isempty(summaryFiles)
        matlabSummaryCsv = fullfile(summaryFiles(1).folder, summaryFiles(1).name);
    end
end

manifest.subject_id = inputName;
manifest.input_file = inputFile;
manifest.output_dir = outputDir;
manifest.matlab_output_dir = matlabOutputDir;
manifest.summary_csv = summaryCsv;
manifest.aperiodic_csv = aperiodicCsv;
manifest.fit_csv = fitCsv;
manifest.matlab_summary_csv = matlabSummaryCsv;
manifest.freq_range = [fmin fmax];
manifest.save_fooof_img = saveFooofImg;
manifest.parallel = useParallel;
manifest.n_channels = EEG.nbchan;
manifest.n_points = EEG.pnts;
manifest.n_epochs = EEG.trials;
manifest.sampling_rate = EEG.srate;
manifest.summary_row_count = height(summary);
manifest.aperiodic_row_count = height(aperiodicRows);
manifest.generated_at = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));

manifestPath = fullfile(outputDir, sprintf('%s_fooof_manifest.json', inputName));
fid = fopen(manifestPath, 'w');
if fid == -1
    error('Could not open manifest for writing: %s', manifestPath);
end
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(manifest));
end
