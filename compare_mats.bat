@REM @echo off
@REM @REM matlab -nosplash -nodesktop -r "run('compare.m');"
@REM matlab -nosplash -nodesktop -wait -r "visdiff('sk_automotive_20221003_164605_correct.skv.mat', 'sk_automotive_20221003_164605_out.skv.mat');"

@echo off

REM Check that two arguments were provided
if [%1]==[] goto usage
if [%2]==[] goto usage

REM Get the input arguments
set file1=%1
set file2=%2

REM Run MATLAB and call the visdiff() function with the input file names
matlab -nosplash -nodesktop -wait -r "visdiff('%file1%', '%file2%');"

goto end

:usage
echo Usage: %0 file1.mat file2.mat
echo.
echo Example: %0 input1.mat input2.mat
echo.
echo Note: Both file1.mat and file2.mat must be provided.

:end
