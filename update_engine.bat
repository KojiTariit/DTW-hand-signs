@echo off
echo ===================================================
echo   Signs Sense: HYBRID ENGINE UPDATER
echo ===================================================
echo.
echo 1/5 [CLUSTERING] Organizing signs into neighborhoods...
python Unsupervised_learning.py
if %ERRORLEVEL% NEQ 0 (
    echo !! Clustering Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 2/5 [TRAINING] Re-learning static signs from templates...
python ml_project/train_ml.py



if %ERRORLEVEL% NEQ 0 (
    echo !! Training Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 3/5 [TRAINING] Re-learning dynamic shape signatures...


python ml_project/train_dynamic_ml.py


if %ERRORLEVEL% NEQ 0 (
    echo !! Dynamic Training Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 4/5 [TRANSPILING] Exporting "Knowledge" to C++ Header...


python ml_project/export_cpp.py
python ml_project/export_dynamic_cpp.py


if %ERRORLEVEL% NEQ 0 (
    echo !! Export Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 5/5 [COMPILING] Baking new brain into scrap_receiver.exe...
echo NOTE: Make sure scrap_receiver.exe is CLOSED!

g++ -O2 scrap_receiver.cpp -o scrap_receiver.exe -lws2_32 -Wl,--stack,16777216



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
