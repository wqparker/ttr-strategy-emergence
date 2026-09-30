# Eighth linear pass (PLAN.md "Current progress"): the two ingredients that each helped against
# greedy, combined: lambda 0.98 at alpha 0.2 (pass 7) with alpha decay to 0.02 (pass 6).
# Q against greedy and against self; 2 settings x 5 seeds, 30000 games, all at once.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass8.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass8/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon reaches 0.02 at game 1000 as in passes 2-7; evaluations every 500 games
$common = "--games 30000 --alpha 0.2 --alpha-end 0.02 --lambda 0.98 --average 100 --epsilon-decay 0.0333 --eval-every 500 --live 500"
$runs = [ordered]@{
  "p8a_q_lam98_greedy_decay" = "--algo q --opponent greedy"
  "p8b_q_lam98_self_decay"   = "--algo q --opponent self"
}

Invoke-RunQueue -Exe ttr-train-linear -Out "runs\linear\pass8" -Common $common -Runs $runs -Seeds (0..4) -Slots 12 -Log pass8.log -DryRun:$DryRun
