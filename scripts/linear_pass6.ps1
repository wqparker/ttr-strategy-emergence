# Sixth linear pass (PLAN.md "Current progress"): pass 5's Q-learning peaked near game 5000 and
# then declined against greedy and self-play at constant alpha. Tests alpha decay (0.05 -> 0.005)
# over {Q, SARSA} x {random, greedy, self}, plus SARSA at constant alpha against greedy and self
# (pass 5 has the other constant-alpha cells). 8 settings x 5 seeds, 30000 games, 12 at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass6.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass6/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon reaches 0.02 at game 1000 as in passes 2-5; evaluations every 500 games
$common = "--games 30000 --average 100 --epsilon-decay 0.0333 --eval-every 500 --live 500"
$decay = "--alpha 0.05 --alpha-end 0.005"
$runs = [ordered]@{
  "p6a_q_random_decay"     = "--algo q --opponent random $decay"
  "p6b_sarsa_random_decay" = "--algo sarsa --opponent random $decay"
  "p6c_q_greedy_decay"     = "--algo q --opponent greedy $decay"
  "p6d_sarsa_greedy_decay" = "--algo sarsa --opponent greedy $decay"
  "p6e_q_self_decay"       = "--algo q --opponent self $decay"
  "p6f_sarsa_self_decay"   = "--algo sarsa --opponent self $decay"
  "p6g_sarsa_greedy"       = "--algo sarsa --opponent greedy --alpha 0.05"
  "p6h_sarsa_self"         = "--algo sarsa --opponent self --alpha 0.05"
}

Invoke-RunQueue -Exe ttr-train-linear -Out "runs\linear\pass6" -Common $common -Runs $runs -Seeds (0..4) -Slots 12 -Log pass6.log -DryRun:$DryRun
