% Load the correct file
% file1 = load('sk_automotive_20221003_164605_correct.skv.mat');

% Load the file in question
% file2 = load('sk_automotive_20221003_164605_out.skv.mat');

% Compare the files using visdiff
% visdiff('data1.mat', 'data2.mat');
% visdiff('sk_automotive_20221003_164605_correct.skv.mat', 'sk_automotive_20221003_164605_out.skv.mat');







% % Create an input parser object
% parser = inputParser;

% % Add arguments to the input parser
% addParameter(parser, '-f1', '', @ischar);
% addParameter(parser, '-f2', '', @ischar);

% % Parse the input arguments
% parse(parser, varargin{:});

% % Get the input file names
% file1 = parser.Results.f1;
% file2 = parser.Results.f2;

% % Check that the input files exist
% if ~exist(file1, 'file')
%     error('Input file %s not found.', file1);
% end

% if ~exist(file2, 'file')
%     error('Input file %s not found.', file2);
% end

% % Compare the files
% visdiff(file1, file2);






function compare_mats(file1, file2)
    % COMPARE_MATS - Script to compare two .mat input files using visdiff and display the results
    %
    % Usage:
    %   $ matlab -r "compare_mats('file1.mat', 'file2.mat');"
    %
    % Inputs:
    %   file1 - Name of the first file to compare
    %   file2 - Name of the second file to compare
    %
    % Examples:
    %   $ matlab -r "compare_mats('sk_automotive_20221003_164605_correct.skv.mat', 'sk_automotive_20221003_164605_out.skv.mat');"
    %   $ matlab -nosplash -r "compare_mats('sk_automotive_20221003_164605_correct.skv.mat', 'sk_automotive_20221003_164605_out.skv.mat');"
    %   $ matlab -nosplash -nodesktop -r "compare_mats('sk_automotive_20221003_164605_correct.skv.mat', 'sk_automotive_20221003_164605_out.skv.mat');"
    %   $ matlab -nodesktop -r "compare_mats('sk_automotive_20221003_164605_correct.skv.mat', 'sk_automotive_20221003_164605_out.skv.mat');"
    
    % % Print the input args
    % fprintf('File 1: %s\n', file1);
    % fprintf('File 2: %s\n', file2);

    % Check that the input files exist
    if ~exist(file1, 'file')
        error('Input file %s not found.', file1);
    end
    
    if ~exist(file2, 'file')
        error('Input file %s not found.', file2);
    end
    
    % Compare the files
    visdiff(file1, file2);
    