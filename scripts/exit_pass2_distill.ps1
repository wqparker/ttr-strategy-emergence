# Expert iteration, pass 2, stage 2 (PLAN.md "Current progress"): PPO's network trained, supervised, on the f1d-guided
# search's recorded decisions (stage 1: scripts\exit_pass2.ps1, runs\exit\pass2\data). Every arm observes with the
# ticket plan (pass 1: without it a network can't copy the search).
#   x2a_plan_move    move targets (pass 1's x1e: the start for PPO, which samples)
#   x2b_plan_visits  visit-count targets (pass 1's x1c: the spread a search prior wants)
#   x2c_both_move    x2a on pass 1's and pass 2's data together (pass 1: half the data cost 15-20 points)
# Value head: own score still to come. 40 epochs, evaluated every 5 against random, greedy, wary, racer, collector,
# PPO p2b and linear p7b (100 games each); best checkpoint on the mean margin over all but random and linear. Games
# numbered 9, 19, 29, ... are held out. 3 arms x 3 seeds, 6 at a time, about 30 min.
# -WaitPid PID: first wait for that process (the stage 1 launcher) to exit; stage 1 must have finished.
#   powershell -ExecutionPolicy Bypass -File scripts\exit_pass2_distill.ps1 [-WaitPid PID]   (-DryRun prints the commands)
param([switch]$DryRun, [int]$WaitPid = 0)
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

if (-not $DryRun) {
  if ($WaitPid) {
    Set-KeepAwake $true  # no idle sleep between stage 1's end and the queue's start
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
  }
  if (-not (Select-String -Path "runs\exit\pass2\teacher.log" -Pattern " finished" -Quiet)) {
    throw "stage 1 has not finished (runs\exit\pass2\teacher.log)"
  }
}

$ppo = "ppo:runs/ppo/pass2/p2b_pool_s3.json"
$common = "--epochs 40 --eval-every 5 --ticket-plan --eval-opponents random greedy wary racer collector $ppo"
$runs = [ordered]@{
  "x2a_plan_move"   = "--data runs\exit\pass2\data --target move"
  "x2b_plan_visits" = "--data runs\exit\pass2\data"
  "x2c_both_move"   = "--data runs\exit\pass1\data runs\exit\pass2\data --target move"
}

Invoke-RunQueue -Exe ttr-train-distill -Out "runs\exit\pass2" -Common $common -Runs $runs -Seeds (0..2) -Slots 6 -Log distill.log -DryRun:$DryRun
