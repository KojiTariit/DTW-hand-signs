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
echo 4/5 [TRANSPILING] Exporting Forests to JSON...
python export_forests.py

if %ERRORLEVEL% NEQ 0 (
    echo !! Export Failed !!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 5/5 [COMPILING] Baking new brains into scrap_receiver.exe and continuous_receiver.exe...
echo NOTE: Make sure BOTH receivers are CLOSED!

g++ -O2 scrap_receiver.cpp -o scrap_receiver.exe -lws2_32
g++ -O2 continuous_receiver.cpp -o continuous_receiver.exe -lws2_32

if %ERRORLEVEL% NEQ 0 (
    echo !! Compilation Failed !!
    echo Check if scrap_receiver.exe or continuous_receiver.exe is still open.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo SUCCESS: Engine is now 100%% up to date!
echo ===================================================
pause
