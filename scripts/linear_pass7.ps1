# Seventh linear pass (PLAN.md "Current progress"): pass 4's strongest setting (p4f: lambda 0.98,
# alpha 0.2, average 100; still rising at 10000 games) run longer and against the stronger
# opponents, since re-scored best checkpoints trained against greedy and self beat greedy.
# 4 settings x 5 seeds, 30000 games, at most 12 runs at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass7.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass7/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\linear\pass7"
$slots = 12
New-Item -ItemType Directory -Force $out | Out-Null

# epsilon reaches 0.02 at game 1000 as in passes 2-6; evaluations every 500 games
$common = "--games 30000 --alpha 0.2 --lambda 0.98 --average 100 --epsilon-decay 0.0333 --eval-every 500 --live 500"
$runs = [ordered]@{
  "p7a_q_lam98_random"     = "--algo q --opponent random"      # p4f, longer
  "p7b_sarsa_lam98_random" = "--algo sarsa --opponent random"  # SARSA(lambda)
  "p7c_q_lam98_greedy"     = "--algo q --opponent greedy"
  "p7d_q_lam98_self"       = "--algo q --opponent self"
}
$queue = [System.Collections.Queue]::new()
foreach ($s in 0..4) { foreach ($name in $runs.Keys) { $queue.Enqueue(@($name, $s)) } }

Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Out-File -Encoding utf8 "$out\pass7.log"
$jobs = @()
while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
  while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $slots) {
    $name, $s = $queue.Dequeue()
    $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-linear.exe `
      -ArgumentList "$common $($runs[$name]) --seed $s --out $out\${name}_s$s.json" `
      -RedirectStandardOutput "$out\${name}_s$s.log" -RedirectStandardError "$out\${name}_s$s.err"
    $null = $proc.Handle  # keep the handle, or ExitCode reads empty after exit
    $jobs += [pscustomobject]@{ Run = "${name}_s$s"; Proc = $proc }
    "start $(Get-Date -Format s)  ${name}_s$s" | Out-File -Append -Encoding utf8 "$out\pass7.log"
  }
  Start-Sleep -Seconds 10
}
foreach ($j in $jobs) { "exit $($j.Proc.ExitCode)  $($j.Run)" | Out-File -Append -Encoding utf8 "$out\pass7.log" }
"finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$out\pass7.log"

[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
