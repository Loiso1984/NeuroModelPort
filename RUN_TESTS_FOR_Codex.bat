@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM -----------------------------------------------------------------------------
REM Codex-requested test runner for Windows
REM Project root expected at: C:\NeuroModelPort
REM -----------------------------------------------------------------------------

set "ROOT=C:\NeuroModelPort"
cd /d "%ROOT%" || (
  echo [FATAL] Cannot cd to %ROOT%
  exit /b 1
)

set "LOGDIR=%ROOT%\tests\artifacts\codex_requested_runs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "GATE_STEP_TIMEOUT_SEC=1500"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOG=%LOGDIR%\run_%TS%.log"
set "FAIL_COUNT=0"

echo ====================================================== > "%LOG%"
echo Codex requested test run started at %DATE% %TIME% >> "%LOG%"
echo Root: %ROOT% >> "%LOG%"
echo ====================================================== >> "%LOG%"

echo.
echo [INFO] Logging to: %LOG%

call :run_step "Core fast suite" "python -m pytest -q tests/core/test_p0_p1_gate_runner.py tests/core/test_run_f_conduction_extended_cli.py tests/core/test_preset_stress_validation_cli.py tests/core/test_jacobian_contract.py tests/core/test_dual_stimulation_distribution.py tests/core/test_delay_target_utils.py tests/core/test_unit_converter_current.py"
call :run_step "Python syntax preflight" "python -m compileall -q core gui tests"
call :run_step "MS branch attenuation test" "python -m pytest -q tests/branches/test_ms_conduction_block_branch.py"
call :run_step "Consolidated P0/P1 gate" "python tests/utils/run_p0_p1_gate.py --out-dir tests/artifacts/p0_p1_gate_user --target-ratio 0.3 --step-timeout-sec %GATE_STEP_TIMEOUT_SEC%"
call :run_step "F conduction diagnostic sweep" "python tests/utils/run_f_conduction_extended.py --target-ratio 0.3 --no-fail-on-anomaly --output tests/artifacts/f_conduction_user.json"
call :run_step "Preset stress (bounded runtime)" "python tests/utils/run_preset_stress_validation.py --limit-presets 8 --dt-eval 0.2 --no-fail-on-fail --out tests/artifacts/preset_stress_user.json --report-md tests/artifacts/preset_stress_user.md"

echo. >> "%LOG%"
echo ====================================================== >> "%LOG%"
echo Finished at %DATE% %TIME% >> "%LOG%"
echo FAIL_COUNT=%FAIL_COUNT% >> "%LOG%"
echo ====================================================== >> "%LOG%"

echo.
echo [INFO] Finished. FAIL_COUNT=%FAIL_COUNT%
echo [INFO] Share these files with Codex:
echo   1) %LOG%
echo   2) tests\artifacts\p0_p1_gate_user\p0_p1_gate_summary.json
echo   3) tests\artifacts\p0_p1_gate_user\p0_p1_gate_summary.md
echo   4) tests\artifacts\f_conduction_user.json
echo   5) tests\artifacts\preset_stress_user.json
echo   6) tests\artifacts\preset_stress_user.md

if not "%FAIL_COUNT%"=="0" exit /b 1
exit /b 0

:run_step
set "STEP_NAME=%~1"
set "STEP_CMD=%~2"

echo. >> "%LOG%"
echo ------------------------------------------------------ >> "%LOG%"
echo [STEP] %STEP_NAME% >> "%LOG%"
echo [CMD ] %STEP_CMD% >> "%LOG%"
echo ------------------------------------------------------ >> "%LOG%"

echo.
echo [RUN] %STEP_NAME%
cmd /c "%STEP_CMD%" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

if not "!RC!"=="0" (
  set /a FAIL_COUNT+=1
  echo [FAIL] %STEP_NAME% (exit=!RC!)
  echo [FAIL] %STEP_NAME% (exit=!RC!) >> "%LOG%"
) else (
  echo [PASS] %STEP_NAME%
  echo [PASS] %STEP_NAME% >> "%LOG%"
)
exit /b 0
