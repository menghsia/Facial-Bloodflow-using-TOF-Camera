@echo off

cmake -G "Visual Studio 14 2015 Win64" -S . -B .build/build
cmake --build .build/build --target ALL_BUILD --config Release

echo.
echo.
echo Visual Studio solution generated in:  .build/build/imx520_sample.sln
echo Executable generated in:              .build/output/bin/Release/imx520_sample.exe