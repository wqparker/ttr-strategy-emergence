# Second DQN pass (PLAN.md "Current progress", tier B): does an opponent pool break the racing
# strategy every learner found against greedy, and does ticket shaping bring ticket play? One-step
# Double DQN (pass 1: n = 1 beat n = 8 and 32), averaging 100. Arms: vs greedy (pass-1 reference),
# vs the pool (greedy, wary, racer, self), vs the pool with shaping 1. 3 x 3 seeds, 30000 games.
#   powershell -ExecutionPolicy Bypass -File scripts\dqn_pass2.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/dqn/pass2/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/dqn/pass2/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\dqn\pass2"
$slots = 9
New-Item -ItemType Directory -Force $out | Out-Null

# epsilon 1.0 -> 0.02 over the first 1500 games; evaluations every 1000 games (100 games each vs
# random, greedy, wary, racer and the best linear agent); best checkpoint on the mean over greedy, wary, racer
$common = "--games 30000 --n-step 1 --average 100 --eval-every 1000 --live 1000"
$runs = [ordered]@{
  "d2a_greedy"     = "--opponent greedy"
  "d2b_pool"       = "--opponent pool"
  "d2c_pool_shape" = "--opponent pool --shaping 1"
}
$queue = [System.Collections.Queue]::new()
foreach ($s in 0..2) { foreach ($name in $runs.Keys) { $queue.Enqueue(@($name, $s)) } }

# Keep the PC awake while this script runs (released at the end).
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Out-File -Encoding utf8 "$out\pass2.log"
$jobs = @()
while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
  while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $slots) {
    $name, $s = $queue.Dequeue()
    $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-dqn.exe `
      -ArgumentList "$common $($runs[$name]) --seed $s --out $out\${name}_s$s.json" `
      -RedirectStandardOutput "$out\${name}_s$s.log" -RedirectStandardError "$out\${name}_s$s.err"
    $null = $proc.Handle  # keep the handle, or ExitCode reads empty after exit
    $jobs += [pscustomobject]@{ Run = "${name}_s$s"; Proc = $proc }
    "start $(Get-Date -Format s)  ${name}_s$s" | Out-File -Append -Encoding utf8 "$out\pass2.log"
  }
  Start-Sleep -Seconds 10
}
foreach ($j in $jobs) { "exit $($j.Proc.ExitCode)  $($j.Run)" | Out-File -Append -Encoding utf8 "$out\pass2.log" }
"finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$out\pass2.log"

[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
