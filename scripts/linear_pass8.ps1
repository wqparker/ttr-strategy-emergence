# Eighth linear pass (PLAN.md "Current progress"): the two ingredients that each helped against
# greedy, combined: lambda 0.98 at alpha 0.2 (pass 7) with alpha decay to 0.02 (pass 6).
# Q against greedy and against self; 2 settings x 5 seeds, 30000 games, all at once.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass8.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass8/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\linear\pass8"
$slots = 12
New-Item -ItemType Directory -Force $out | Out-Null

# epsilon reaches 0.02 at game 1000 as in passes 2-7; evaluations every 500 games
$common = "--games 30000 --alpha 0.2 --alpha-end 0.02 --lambda 0.98 --average 100 --epsilon-decay 0.0333 --eval-every 500 --live 500"
$runs = [ordered]@{
  "p8a_q_lam98_greedy_decay" = "--algo q --opponent greedy"
  "p8b_q_lam98_self_decay"   = "--algo q --opponent self"
}
$queue = [System.Collections.Queue]::new()
foreach ($s in 0..4) { foreach ($name in $runs.Keys) { $queue.Enqueue(@($name, $s)) } }

Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Out-File -Encoding utf8 "$out\pass8.log"
$jobs = @()
while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
  while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $slots) {
    $name, $s = $queue.Dequeue()
    $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-linear.exe `
      -ArgumentList "$common $($runs[$name]) --seed $s --out $out\${name}_s$s.json" `
      -RedirectStandardOutput "$out\${name}_s$s.log" -RedirectStandardError "$out\${name}_s$s.err"
    $null = $proc.Handle  # keep the handle, or ExitCode reads empty after exit
    $jobs += [pscustomobject]@{ Run = "${name}_s$s"; Proc = $proc }
    "start $(Get-Date -Format s)  ${name}_s$s" | Out-File -Append -Encoding utf8 "$out\pass8.log"
  }
  Start-Sleep -Seconds 10
}
foreach ($j in $jobs) { "exit $($j.Proc.ExitCode)  $($j.Run)" | Out-File -Append -Encoding utf8 "$out\pass8.log" }
"finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$out\pass8.log"

[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
