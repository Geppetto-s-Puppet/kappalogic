@echo off
REM Build the CUDA SP2 demo. Requires: NVIDIA GPU, CUDA Toolkit (nvcc), and MSVC.
REM Adjust the vcvars64.bat path below to your Visual Studio installation.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
REM /wd4819 silences the CUDA-header codepage warning on non-UTF8 (e.g. Japanese) locales.
nvcc -O2 -Xcompiler "/wd4819 /EHsc" -o sp2_cuda.exe sp2_cuda.cu -lcublas
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
echo BUILD OK  ^-^-^>  run:  sp2_cuda.exe 2048 1024
