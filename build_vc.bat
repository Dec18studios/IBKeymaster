@echo off
setlocal
set ROOT=%~dp0
set CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set VCVARS_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%CMAKE_EXE%" (
  echo Expected CMake at "%CMAKE_EXE%" but it was not found.
  exit /b 1
)
if not exist "%VCVARS_EXE%" (
  echo Expected vcvars64 at "%VCVARS_EXE%" but it was not found.
  exit /b 1
)
call "%ROOT%configure_vc.bat" || exit /b 1
call "%VCVARS_EXE%" >nul
"%CMAKE_EXE%" --build --preset windows-release %*
