@echo off
echo ========================================
echo Installing Dependencies for Phase A
echo ========================================

echo Installing core packages...
pip install numpy pandas pillow -q
if errorlevel 1 goto error

echo Installing PyTorch (this may take a few minutes)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
if errorlevel 1 goto error

echo Installing computer vision packages...
pip install opencv-python -q
if errorlevel 1 goto error

echo Installing ML tools...
pip install scikit-learn tqdm -q
if errorlevel 1 goto error

echo Installing augmentation libraries...
pip install albumentations -q
if errorlevel 1 goto error

echo Installing timm...
pip install timm -q
if errorlevel 1 goto error

echo Installing tensorboard...
pip install tensorboard -q
if errorlevel 1 goto error

echo.
echo ========================================
echo SUCCESS! All dependencies installed.
echo ========================================
echo.
echo You can now run:
echo   python train_task1_binary.py --data_path data/train/ --epochs 15 --batch_size 4
echo.
goto end

:error
echo.
echo ========================================
echo ERROR: Installation failed!
echo ========================================
echo Try closing other Python processes and run again.
echo.

:end
pause
