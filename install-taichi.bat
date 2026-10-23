@echo off

set "python_exec=..\..\..\python_embeded\python.exe"
echo Installing Taichi lang backend...

if exist "%python_exec%" (
    %python_exec% -s -m pip install taichi
) else (
    echo Installing with system Python
    pip install taichi
)

pause