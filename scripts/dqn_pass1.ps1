# First DQN pass (PLAN.md "Current progress", tier B): Double DQN with a dueling head against
# greedy, the setting the linear learner did best in, with weight averaging (tier A's clearest
# gain) on. The question is credit assignment: one-step targets vs n-step returns, the DQN
# counterpart of lambda. 3 settings x 3 seeds, 30000 games, all at once.
#   powershell -ExecutionPolicy Bypass -File scripts\dqn_pass1.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/dqn/pass1/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/dqn/pass1/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon 1.0 -> 0.02 over the first 1500 games; evaluations every 500 games (100 games each vs
# random, greedy and the best linear agent), the same paired games as the linear passes
$common = "--games 30000 --opponent greedy --average 100 --eval-every 500 --live 500"
$runs = [ordered]@{
  "d1a_n1"  = "--n-step 1"
  "d1b_n8"  = "--n-step 8"
  "d1c_n32" = "--n-step 32"
}

Invoke-RunQueue -Exe ttr-train-dqn -Out "runs\dqn\pass1" -Common $common -Runs $runs -Seeds (0..2) -Slots 9 -Log pass1.log -DryRun:$DryRun
