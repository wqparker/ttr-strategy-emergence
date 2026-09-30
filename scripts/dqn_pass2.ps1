# Second DQN pass (PLAN.md "Current progress", tier B): does an opponent pool break the racing
# strategy every learner found against greedy, and does ticket shaping bring ticket play? One-step
# Double DQN (pass 1: n = 1 beat n = 8 and 32), averaging 100. Arms: vs greedy (pass-1 reference),
# vs the pool (greedy, wary, racer, self), vs the pool with shaping 1. 3 x 3 seeds, 30000 games.
#   powershell -ExecutionPolicy Bypass -File scripts\dqn_pass2.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/dqn/pass2/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/dqn/pass2/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon 1.0 -> 0.02 over the first 1500 games; evaluations every 1000 games (100 games each vs
# random, greedy, wary, racer and the best linear agent); best checkpoint on the mean over greedy, wary, racer
$common = "--games 30000 --n-step 1 --average 100 --eval-every 1000 --live 1000"
$runs = [ordered]@{
  "d2a_greedy"     = "--opponent greedy"
  "d2b_pool"       = "--opponent pool"
  "d2c_pool_shape" = "--opponent pool --shaping 1"
}

Invoke-RunQueue -Exe ttr-train-dqn -Out "runs\dqn\pass2" -Common $common -Runs $runs -Seeds (0..2) -Slots 9 -Log pass2.log -DryRun:$DryRun
