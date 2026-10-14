# Expert iteration, pass 1, stage 2 (PLAN.md "Current progress"): PPO's network trained, supervised, on the ticket
# search's recorded decisions (stage 1: scripts\exit_pass1.ps1, runs\exit\pass1\data). Does the network play tickets?
#   x1a_visits   policy target: the search's root visit counts (AlphaZero's); plain observation
#   x1b_move     policy target: the search's move only
#   x1c_plan     x1a + the ticket-plan observation block
#   x1d_half     x1a on the first 250 games against each opponent: would more data help?
# Value head: own score still to come (the teacher's reward). 40 epochs, evaluated every 5 against random, greedy,
# wary, racer, collector, PPO p2b and linear p7b (100 games each); best checkpoint on the mean margin over all but
# random and linear. Games numbered 9, 19, 29, ... are held out for validation. 4 arms x 3 seeds, 6 at a time.
# -WaitPid PID: first wait for that process (the stage 1 launcher) to exit; stage 1 must have finished.
#   powershell -ExecutionPolicy Bypass -File scripts\exit_pass1_distill.ps1 [-WaitPid PID]   (-DryRun prints the commands)
param([switch]$DryRun, [int]$WaitPid = 0)
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

if (-not $DryRun) {
  if ($WaitPid) {
    Set-KeepAwake $true  # no idle sleep between stage 1's end and the queue's start
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
  }
  if (-not (Select-String -Path "runs\exit\pass1\teacher.log" -Pattern " finished" -Quiet)) {
    throw "stage 1 has not finished (runs\exit\pass1\teacher.log)"
  }
}

$ppo = "ppo:runs/ppo/pass2/p2b_pool_s3.json"
$common = "--data runs\exit\pass1\data --epochs 40 --eval-every 5 --eval-opponents random greedy wary racer collector $ppo"
$runs = [ordered]@{
  "x1a_visits" = ""
  "x1b_move"   = "--target move"
  "x1c_plan"   = "--ticket-plan"
  "x1d_half"   = "--first-games 250"
}

Invoke-RunQueue -Exe ttr-train-distill -Out "runs\exit\pass1" -Common $common -Runs $runs -Seeds (0..2) -Slots 6 -Log distill.log -DryRun:$DryRun
