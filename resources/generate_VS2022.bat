@echo off

cmake -G "Visual Studio 17 2022" -A x64 -S . -B .build/build
cmake --build .build/build --config Release

echo.
echo.
echo Visual Studio solution generated in:  .build/build/imx520_sample.sln
echo Executable generated in:              .build/output/bin/Release/imx520_sample.exe

PAUSE