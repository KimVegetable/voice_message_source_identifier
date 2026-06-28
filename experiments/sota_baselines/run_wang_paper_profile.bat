@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0..\..\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
if not exist "results" mkdir "results"
"%PYTHON%" "run_baselines.py" --method wang_proposed --protocol logo --sample-rate 32000 --wang-tune-profile paper --output-dir "results" 1> "results\wang_proposed_paper_stdout.log" 2> "results\wang_proposed_paper_stderr.log"
