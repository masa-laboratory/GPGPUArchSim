@echo off
echo Starting the Sphinx automated build process...
echo source directory: source
echo target directory: build/html

:: Run the sphinx-autobuild command
sphinx-autobuild source build/html

:: Check whether the command was executed successfully
if %errorlevel% neq 0 (
    echo Sphinx auto-build process encountered an error. Please check the error messages.
) else (
    echo Sphinx auto-build process completed successfully.
)

pause
