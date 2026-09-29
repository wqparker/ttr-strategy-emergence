# DQN trained against random only (a side experiment, 2026-09-29): what does a learner find when its only
# opponent competes for nothing? Same settings as pass 2's greedy arm (n = 1, averaging 100, no shaping),
# so the two differ only in the opponent. Evaluated against every bot and the held-out linear agent; judge
# it by the final weights (the best checkpoint is picked on the bots). 3 seeds, 30000 games.
#   powershell -ExecutionPolicy Bypass -File scripts\dqn_random.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/dqn/random/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/dqn/random/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\dqn\random"
$slots = 3
New-Item -ItemType Directory -Force $out | Out-Null

# epsilon 1.0 -> 0.02 over the first 1500 games; evaluations every 1000 games (100 games each vs
# random, greedy, wary, racer, collector and the best linear agent)
$common = "--games 30000 --n-step 1 --average 100 --opponent random --eval-opponents random greedy wary racer collector --eval-every 1000 --live 1000"
$runs = [ordered]@{
  "dr_random" = ""
}
$queue = [System.Collections.Queue]::new()
foreach ($s in 0..2) { foreach ($name in $runs.Keys) { $queue.Enqueue(@($name, $s)) } }

# Keep the PC awake while this script runs (released at the end).
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Out-File -Encoding utf8 "$out\random.log"
$jobs = @()
while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
  while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $slots) {
    $name, $s = $queue.Dequeue()
    $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-dqn.exe `
      -ArgumentList "$common $($runs[$name]) --seed $s --out $out\${name}_s$s.json" `
      -RedirectStandardOutput "$out\${name}_s$s.log" -RedirectStandardError "$out\${name}_s$s.err"
    $null = $proc.Handle  # keep the handle, or ExitCode reads empty after exit
    $jobs += [pscustomobject]@{ Run = "${name}_s$s"; Proc = $proc }
    "start $(Get-Date -Format s)  ${name}_s$s" | Out-File -Append -Encoding utf8 "$out\random.log"
  }
  Start-Sleep -Seconds 10
}
foreach ($j in $jobs) { "exit $($j.Proc.ExitCode)  $($j.Run)" | Out-File -Append -Encoding utf8 "$out\random.log" }
"finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$out\random.log"

[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
