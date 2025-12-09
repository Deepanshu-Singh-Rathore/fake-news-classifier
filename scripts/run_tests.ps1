param (
    [string]$TestPath = "tests/test_pipeline.py"
)

# Try to use local venv python if available
$venvPython = Join-Path -Path (Join-Path -Path $PSScriptRoot -ChildPath "..") -ChildPath ".venv/Scripts/python.exe"
if (Test-Path $venvPython) {
    & $venvPython -m pytest -q $TestPath
} else {
    python -m pytest -q $TestPath
}
