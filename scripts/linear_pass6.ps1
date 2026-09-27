# Sixth linear pass (PLAN.md "Current progress"): pass 5's Q-learning peaked near game 5000 and
# then declined against greedy and self-play at constant alpha. Tests alpha decay (0.05 -> 0.005)
# over {Q, SARSA} x {random, greedy, self}, plus SARSA at constant alpha against greedy and self
# (pass 5 has the other constant-alpha cells). 8 settings x 5 seeds, 30000 games, 12 at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass6.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass6/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\linear\pass6"
$slots = 12
New-Item -ItemType Directory -Force $out | Out-Null

# epsilon reaches 0.02 at game 1000 as in passes 2-5; evaluations every 500 games
$common = "--games 30000 --average 100 --epsilon-decay 0.0333 --eval-every 500 --live 500"
$decay = "--alpha 0.05 --alpha-end 0.005"
$runs = [ordered]@{
  "p6a_q_random_decay"     = "--algo q --opponent random $decay"
  "p6b_sarsa_random_decay" = "--algo sarsa --opponent random $decay"
  "p6c_q_greedy_decay"     = "--algo q --opponent greedy $decay"
  "p6d_sarsa_greedy_decay" = "--algo sarsa --opponent greedy $decay"
  "p6e_q_self_decay"       = "--algo q --opponent self $decay"
  "p6f_sarsa_self_decay"   = "--algo sarsa --opponent self $decay"
  "p6g_sarsa_greedy"       = "--algo sarsa --opponent greedy --alpha 0.05"
  "p6h_sarsa_self"         = "--algo sarsa --opponent self --alpha 0.05"
}
$queue = [System.Collections.Queue]::new()
foreach ($s in 0..4) { foreach ($name in $runs.Keys) { $queue.Enqueue(@($name, $s)) } }

Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Out-File -Encoding utf8 "$out\pass6.log"
$jobs = @()
while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
  while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $slots) {
    $name, $s = $queue.Dequeue()
    $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-linear.exe `
      -ArgumentList "$common $($runs[$name]) --seed $s --out $out\${name}_s$s.json" `
      -RedirectStandardOutput "$out\${name}_s$s.log" -RedirectStandardError "$out\${name}_s$s.err"
    $null = $proc.Handle  # keep the handle, or ExitCode reads empty after exit
    $jobs += [pscustomobject]@{ Run = "${name}_s$s"; Proc = $proc }
    "start $(Get-Date -Format s)  ${name}_s$s" | Out-File -Append -Encoding utf8 "$out\pass6.log"
  }
  Start-Sleep -Seconds 10
}
foreach ($j in $jobs) { "exit $($j.Proc.ExitCode)  $($j.Run)" | Out-File -Append -Encoding utf8 "$out\pass6.log" }
"finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$out\pass6.log"

[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
