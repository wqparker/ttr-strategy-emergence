# DQN trained against random only (a side experiment, 2026-09-29): what does a learner find when its only
# opponent competes for nothing? Same settings as pass 2's greedy arm (n = 1, averaging 100, no shaping),
# so the two differ only in the opponent. Evaluated against every bot and the held-out linear agent; judge
# it by the final weights (the best checkpoint is picked on the bots). 3 seeds, 30000 games.
#   powershell -ExecutionPolicy Bypass -File scripts\dqn_random.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/dqn/random/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/dqn/random/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon 1.0 -> 0.02 over the first 1500 games; evaluations every 1000 games (100 games each vs
# random, greedy, wary, racer, collector and the best linear agent)
$common = "--games 30000 --n-step 1 --average 100 --opponent random --eval-opponents random greedy wary racer collector --eval-every 1000 --live 1000"
$runs = [ordered]@{
  "dr_random" = ""
}

Invoke-RunQueue -Exe ttr-train-dqn -Out "runs\dqn\random" -Common $common -Runs $runs -Seeds (0..2) -Slots 3 -Log random.log -DryRun:$DryRun
