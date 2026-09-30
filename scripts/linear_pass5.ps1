# Fifth linear pass (PLAN.md "Current progress"): pass 4's p4b setting (alpha 0.05, average 100)
# across training opponents and Q vs SARSA; 5 settings x 5 seeds, 30000 games each, at most
# 12 runs at a time. Keeps the machine awake until every run ends.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass5.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass5/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon reaches 0.02 at game 1000 as in passes 2-4; evaluations every 500 games
$common = "--games 30000 --alpha 0.05 --average 100 --epsilon-decay 0.0333 --eval-every 500 --live 500"
$runs = [ordered]@{
  "p5a_q_random"     = "--algo q --opponent random"      # p4b, longer: the reference
  "p5b_sarsa_random" = "--algo sarsa --opponent random"  # method comparison
  "p5c_q_greedy"     = "--algo q --opponent greedy"
  "p5d_q_mixed"      = "--algo q --opponent mixed"
  "p5e_q_self"       = "--algo q --opponent self"        # greedy or its own best checkpoint
}

Invoke-RunQueue -Exe ttr-train-linear -Out "runs\linear\pass5" -Common $common -Runs $runs -Seeds (0..4) -Slots 12 -Log pass5.log -DryRun:$DryRun
