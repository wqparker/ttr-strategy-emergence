# First DQN pass (PLAN.md "Current progress", tier B): Double DQN with a dueling head against
# greedy, the setting the linear learner did best in, with weight averaging (tier A's clearest
# gain) on. The question is credit assignment: one-step targets vs n-step returns, the DQN
# counterpart of lambda. 3 settings x 3 seeds, 30000 games, all at once.
#   powershell -ExecutionPolicy Bypass -File scripts\dqn_pass1.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/dqn/pass1/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/dqn/pass1/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\dqn\pass1"
$slots = 9
New-Item -ItemType Directory -Force $out | Out-Null

# epsilon 1.0 -> 0.02 over the first 1500 games; evaluations every 500 games (100 games each vs
# random, greedy and the best linear agent), the same paired games as the linear passes
$common = "--games 30000 --opponent greedy --average 100 --eval-every 500 --live 500"
$runs = [ordered]@{
  "d1a_n1"  = "--n-step 1"
  "d1b_n8"  = "--n-step 8"
  "d1c_n32" = "--n-step 32"
}
$queue = [System.Collections.Queue]::new()
foreach ($s in 0..2) { foreach ($name in $runs.Keys) { $queue.Enqueue(@($name, $s)) } }

# Keep the PC awake while this script runs (released at the end).
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Out-File -Encoding utf8 "$out\pass1.log"
$jobs = @()
while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
  while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $slots) {
    $name, $s = $queue.Dequeue()
    $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-dqn.exe `
      -ArgumentList "$common $($runs[$name]) --seed $s --out $out\${name}_s$s.json" `
      -RedirectStandardOutput "$out\${name}_s$s.log" -RedirectStandardError "$out\${name}_s$s.err"
    $null = $proc.Handle  # keep the handle, or ExitCode reads empty after exit
    $jobs += [pscustomobject]@{ Run = "${name}_s$s"; Proc = $proc }
    "start $(Get-Date -Format s)  ${name}_s$s" | Out-File -Append -Encoding utf8 "$out\pass1.log"
  }
  Start-Sleep -Seconds 10
}
foreach ($j in $jobs) { "exit $($j.Proc.ExitCode)  $($j.Run)" | Out-File -Append -Encoding utf8 "$out\pass1.log" }
"finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$out\pass1.log"

[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
