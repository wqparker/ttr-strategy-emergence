# Seventh linear pass (PLAN.md "Current progress"): pass 4's strongest setting (p4f: lambda 0.98,
# alpha 0.2, average 100; still rising at 10000 games) run longer and against the stronger
# opponents, since re-scored best checkpoints trained against greedy and self beat greedy.
# 4 settings x 5 seeds, 30000 games, at most 12 runs at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass7.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass7/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon reaches 0.02 at game 1000 as in passes 2-6; evaluations every 500 games
$common = "--games 30000 --alpha 0.2 --lambda 0.98 --average 100 --epsilon-decay 0.0333 --eval-every 500 --live 500"
$runs = [ordered]@{
  "p7a_q_lam98_random"     = "--algo q --opponent random"      # p4f, longer
  "p7b_sarsa_lam98_random" = "--algo sarsa --opponent random"  # SARSA(lambda)
  "p7c_q_lam98_greedy"     = "--algo q --opponent greedy"
  "p7d_q_lam98_self"       = "--algo q --opponent self"
}

Invoke-RunQueue -Exe ttr-train-linear -Out "runs\linear\pass7" -Common $common -Runs $runs -Seeds (0..4) -Slots 12 -Log pass7.log -DryRun:$DryRun
