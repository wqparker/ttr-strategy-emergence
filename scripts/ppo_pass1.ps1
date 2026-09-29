# First PPO pass (PLAN.md "Current progress", tier C): DQN pass 3's setting with PPO, 50000 games (PPO
# needs more samples: at 3000 games it was at -70 to -84 vs greedy where DQN was near -45). The pool:
# greedy, wary, racer twice, collector, self, and two strong linear agents; shaping 1; default PPO settings.
# Arms: without and with the observation's ticket-plan block. 2 x 4 seeds, 50000 games, 8 at once.
#   powershell -ExecutionPolicy Bypass -File scripts\ppo_pass1.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/ppo/pass1/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/ppo/pass1/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\ppo\pass1"
$slots = 8
New-Item -ItemType Directory -Force $out | Out-Null

# evaluations every 1000 games (100 games each vs
# random, greedy, wary, racer, collector and the held-out best linear agent p7b); best checkpoint on the
# mean over greedy, wary, racer, collector
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 50000 --opponent pool --pool $pool --shaping 1 --eval-opponents random greedy wary racer collector --eval-every 1000 --live 1000"
$runs = [ordered]@{
  "p1a_pool"      = ""
  "p1b_pool_plan" = "--ticket-plan"
}
$queue = [System.Collections.Queue]::new()
foreach ($s in 0..3) { foreach ($name in $runs.Keys) { $queue.Enqueue(@($name, $s)) } }

# Keep the PC awake while this script runs (released at the end).
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Out-File -Encoding utf8 "$out\pass1.log"
$jobs = @()
while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
  while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $slots) {
    $name, $s = $queue.Dequeue()
    $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-ppo.exe `
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
