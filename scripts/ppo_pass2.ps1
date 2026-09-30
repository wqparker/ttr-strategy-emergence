# Second PPO pass (PLAN.md "Current progress", tier C). Pass 1 was still improving when its learning
# rate reached 0 at 50000 games, and shaping may not be needed with PPO. Pass 1's pool arm (same pool,
# default PPO settings, no ticket plan) for 100000 games, the learning rate annealed over all of them.
# Arms: with shaping 1 (pass 1, longer) and without shaping. 2 x 4 seeds, 8 at once, about 5 h.
#   powershell -ExecutionPolicy Bypass -File scripts\ppo_pass2.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/ppo/pass2/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/ppo/pass2/*.json"
Set-Location (Split-Path $PSScriptRoot)
$out = "runs\ppo\pass2"
$slots = 8
New-Item -ItemType Directory -Force $out | Out-Null

# evaluations every 2000 games (100 games each vs
# random, greedy, wary, racer, collector and the held-out best linear agent p7b); best checkpoint on the
# mean over greedy, wary, racer, collector
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 100000 --opponent pool --pool $pool --eval-opponents random greedy wary racer collector --eval-every 2000 --live 2000"
$runs = [ordered]@{
  "p2a_pool_shape" = "--shaping 1"
  "p2b_pool"       = ""
}
$queue = [System.Collections.Queue]::new()
foreach ($s in 0..3) { foreach ($name in $runs.Keys) { $queue.Enqueue(@($name, $s)) } }

# Keep the PC awake while this script runs (released at the end).
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
  '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

"started $(Get-Date -Format s)" | Out-File -Encoding utf8 "$out\pass2.log"
$jobs = @()
while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
  while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $slots) {
    $name, $s = $queue.Dequeue()
    $proc = Start-Process -NoNewWindow -PassThru -FilePath .venv\Scripts\ttr-train-ppo.exe `
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
