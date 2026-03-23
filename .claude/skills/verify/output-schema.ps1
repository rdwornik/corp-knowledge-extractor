# CKE output schema validator
# Source of truth: corp-os-meta schema + Council Decisions #7, #8
# Last validated: 2026-03-23
#
# Validates that CKE extraction output has all required fields

param([string]$OutputPath)

if (-not $OutputPath) {
    Write-Host "Usage: .\output-schema.ps1 -OutputPath <path_to_extraction_output>"
    exit 1
}

$mds = Get-ChildItem $OutputPath -Recurse -Filter "*.md" -File |
    Where-Object { $_.Name -ne "synthesis.md" -and $_.Name -ne "index.md" -and $_.Directory.Name -eq "extract" }

$totalFiles = $mds.Count
$errors = @()
$warnings = @()

foreach ($md in $mds) {
    $content = Get-Content $md.FullName -Raw -ErrorAction SilentlyContinue
    $name = $md.Name

    # Required fields (Council #7 + #8)
    $required = @{
        "tags:" = "tags field (Council #7)"
        "source_path:" = "source_path"
        "source_hash:" = "source_hash"
        "products:" = "products"
        "extraction_version:" = "extraction_version"
        "routing_reason:" = "routing_reason (Council #8)"
        "prompt_version:" = "prompt_version (Council #8)"
    }

    foreach ($field in $required.GetEnumerator()) {
        if ($content -notmatch "(?m)^$($field.Key)") {
            $errors += "FAIL: $name missing $($field.Value)"
        }
    }

    # Filename convention (Council #10)
    if ($name -notmatch "^\d{4}-\d{2}-\d{2}_.*_[a-f0-9]{4}\.md$") {
        $warnings += "WARN: $name doesn't match naming convention (YYYY-MM-DD_name_hash.md)"
    }

    # Tag format check
    if ($content -match "(?m)^tags:") {
        if ($content -notmatch "product/|topic/|domain/|type/|source/|client/") {
            $warnings += "WARN: $name has tags but no recognized prefix (product/, topic/, domain/, etc.)"
        }
    }
}

Write-Host "`n=== CKE Output Schema Validation ===" -ForegroundColor Cyan
Write-Host "Files checked: $totalFiles"
Write-Host "Errors: $($errors.Count)" -ForegroundColor $(if ($errors.Count -eq 0) {"Green"} else {"Red"})
Write-Host "Warnings: $($warnings.Count)" -ForegroundColor $(if ($warnings.Count -eq 0) {"Green"} else {"Yellow"})

if ($errors.Count -gt 0) {
    Write-Host "`nErrors:" -ForegroundColor Red
    $errors | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    if ($errors.Count -gt 20) { Write-Host "  ... and $($errors.Count - 20) more" }
}

if ($warnings.Count -gt 0) {
    Write-Host "`nWarnings:" -ForegroundColor Yellow
    $warnings | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    if ($warnings.Count -gt 10) { Write-Host "  ... and $($warnings.Count - 10) more" }
}
