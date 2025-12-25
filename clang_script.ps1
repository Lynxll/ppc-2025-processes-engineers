$BuildDir = "build"
$ProjectDir = "tasks\kamalagin_a_mat_mult_strassen"

if (Test-Path $BuildDir) {
    rm -r $BuildDir
}

Write-Host ""
Write-Host "-------> Build project for analysis, using Ninja"

mkdir $BuildDir
cd $BuildDir
cmake -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

Write-Host ""
Write-Host "-------> Analyzing..."
Write-Host ""

Get-ChildItem ..\tasks -Recurse -Include *.cpp, *.hpp | Where-Object {
    $_.FullName -like "*$ProjectDir*"
    } | ForEach-Object {
    Write-Host ""
    Write-Host "Analyzing file: $($_.FullName)" -ForegroundColor Cyan
    Write-Host ""
    
    clang-tidy $_.FullName `
        -p . `
        --warnings-as-errors="*" `
        -header-filter="..\$ProjectDir.*" `
        --system-headers
}

cd ..