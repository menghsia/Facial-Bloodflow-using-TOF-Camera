## mx520_sample
Sample client code using iu456 driver for mx520 cameras

# Requirements
- CMake (cmake.org/download)
- dirent.h (must be added to your global includes directory for Visual Studio. Working on automatically including this with CMake.)

# Build
Run the ``generate_VS2022.bat`` file. This will create:
- a Visual Studio 2022 solution (``.build/build/imx520_sample.sln``)
- an executable of the sample code (``.build/output/Release/imx520_sample.exe``)

# Editing Code
Edit code in ``sources/main.cpp``
To debug with Visual Studio:
- Open the .sln solution file
- Right click the "imx520_sample" project
- Click "Set as Startup Project"
Now your edits to main.cpp under the "imx520_sample" project will be reflected when you press "Local Windows Debugger"