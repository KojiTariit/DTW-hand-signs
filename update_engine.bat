@echo off
echo ===================================================
echo   Signs Sense: HYBRID ENGINE UPDATER
echo ===================================================
echo.
echo 1/4 [TRAINING] Re-learning static signs from templates...


python ml_project/train_ml.py



if %ERRORLEVEL% NEQ 0 (
    echo !! Training Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 2/4 [TRAINING] Re-learning dynamic shape signatures...


python ml_project/train_dynamic_ml.py


if %ERRORLEVEL% NEQ 0 (
    echo !! Dynamic Training Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 3/4 [TRANSPILING] Exporting "Knowledge" to C++ Header...


python ml_project/export_cpp.py
python ml_project/export_dynamic_cpp.py


if %ERRORLEVEL% NEQ 0 (
    echo !! Export Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 4/4 [COMPILING] Baking new brain into scrap_receiver.exe...
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
