@echo off
echo ===================================================
echo   Signs Sense: HYBRID ENGINE UPDATER
echo ===================================================
echo.
echo 1/3 [TRAINING] Re-learning static signs from templates...


python ml_project/train_ml.py



if %ERRORLEVEL% NEQ 0 (
    echo !! Training Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 2/3 [TRANSPILING] Exporting "Knowledge" to C++ Header...


python ml_project/export_cpp.py


if %ERRORLEVEL% NEQ 0 (
    echo !! Export Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 3/3 [COMPILING] Baking new brain into scrap_receiver.exe...
echo NOTE: Make sure scrap_receiver.exe is CLOSED!

g++ -O3 scrap_receiver.cpp -o scrap_receiver.exe -lws2_32



if %ERRORLEVEL% NEQ 0 (
    echo !! Compilation Failed !!
    echo Check if scrap_receiver.exe is still open.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo SUCCESS: Engine is now 100%% up to date!
echo ===================================================
pause
