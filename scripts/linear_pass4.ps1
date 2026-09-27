# Fourth linear pass (PLAN.md "Current progress"): 6 settings x 3 seeds, 10000 games each,
# all in parallel. Keeps the machine awake until every run ends, then lets it sleep again.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass4.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass4/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\linear\pass4"
New-Item -ItemType Directory -Force $out | Out-Null

# epsilon reaches 0.02 at game 1000, as in passes 2 and 3, so their first 2000 games compare
$common = "--algo q --opponent random --games 10000 --epsilon-decay 0.1 --live 100"
$runs = [ordered]@{
  "p4a_base"  = "--alpha 0.05"                                  # pass 2's q_random, run longer
  "p4b_avg"   = "--alpha 0.05 --average 100"                    # averaging at pass 2's alpha
  "p4c_slow"  = "--alpha 0.02 --average 100"                    # p3a run longer: slower or worse?
  "p4d_shape" = "--alpha 0.05 --average 100 --shaping 1"        # shaping without lambda
  "p4e_lam90" = "--alpha 0.05 --average 100 --lambda 0.9"       # a shorter trace
  "p4f_lam98" = "--alpha 0.2 --average 100 --lambda 0.98"       # lambda 0.98 with a larger alpha
}

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this script runs
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Tee-Object -FilePath "$out\pass4.log"
$jobs = foreach ($name in $runs.Keys) { foreach ($s in 0..2) {
  $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-linear.exe `
    -ArgumentList "$common $($runs[$name]) --seed $s --out $out\${name}_s$s.json" `
    -RedirectStandardOutput "$out\${name}_s$s.log" -RedirectStandardError "$out\${name}_s$s.err"
  $null = $proc.Handle  # keep the handle, or ExitCode reads empty after exit
  [pscustomobject]@{ Run = "${name}_s$s"; Proc = $proc } } }
$jobs.Proc | Wait-Process
foreach ($j in $jobs) { "exit $($j.Proc.ExitCode)  $($j.Run)" | Tee-Object -Append -FilePath "$out\pass4.log" }
"finished $(Get-Date -Format s)" | Tee-Object -Append -FilePath "$out\pass4.log"

[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
