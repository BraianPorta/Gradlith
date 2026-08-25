$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$dist = Join-Path $root "apps/playground/dist"
$tmp = Join-Path $env:TEMP "gradlith-gh-pages-publish"

Push-Location $root
try {
  $env:GITHUB_PAGES = "true"
  pnpm --filter "@gradlith/playground" build
  Remove-Item Env:GITHUB_PAGES -ErrorAction SilentlyContinue

  if (Test-Path -LiteralPath $tmp) {
    Remove-Item -LiteralPath $tmp -Recurse -Force
  }

  git clone $root $tmp
  Push-Location $tmp
  try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    git checkout gh-pages 2>&1 | Out-Null
    $checkoutCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($checkoutCode -ne 0) {
      git checkout --orphan gh-pages
    }

    Get-ChildItem -LiteralPath $tmp -Force |
      Where-Object { $_.Name -ne ".git" } |
      Remove-Item -Recurse -Force

    Copy-Item -Path (Join-Path $dist "*") -Destination $tmp -Recurse -Force
    New-Item -Path (Join-Path $tmp ".nojekyll") -ItemType File -Force | Out-Null

    git add -A
    git commit -m "Publish playground to GitHub Pages"
    git remote set-url origin "https://github.com/BraianPorta/Gradlith.git"
    git push origin gh-pages
  } finally {
    Pop-Location
  }
} finally {
  Pop-Location
}
