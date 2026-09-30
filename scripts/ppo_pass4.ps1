# Fourth PPO pass (PLAN.md "Current progress", tier C): the first 3- and 4-player games. In 2-player games
# every method and every reward that learned converged on racing (seven of the nine 6-routes, tickets
# abandoned, the game ended early), with no blocking. With 3-4 players the 6-routes run out, more of the
# board is claimed and ticket paths get cut: does racing survive, and do blocking or tickets appear?
# Rewards: margin (my score - the opponents' mean) and score (my points only), as in pass 3: blocking has
# a reason to exist here, and only margin rewards it. Pass 3's setting (no shaping, default PPO settings,
# the same pool), 50000 games; each opponent seat is drawn from the pool on its own; evaluations fill
# every other seat with the evaluation opponent. 2 player counts x 2 rewards x 3 seeds, all 12 at once.
#   powershell -ExecutionPolicy Bypass -File scripts\ppo_pass4.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/ppo/pass4/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/ppo/pass4/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# evaluations every 2000 games (100 games each vs random, greedy, wary, racer, collector and the
# held-out best linear agent p7b, each filling every other seat), reported as score margins against the
# opponents' mean whatever the training reward; best checkpoint on the mean margin over greedy, wary,
# racer, collector
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 50000 --opponent pool --pool $pool --eval-opponents random greedy wary racer collector --eval-every 2000 --live 2000"
$runs = [ordered]@{
  "p4a_3p_margin" = "--players 3 --reward margin"
  "p4b_3p_score"  = "--players 3 --reward score"
  "p4c_4p_margin" = "--players 4 --reward margin"
  "p4d_4p_score"  = "--players 4 --reward score"
}

Invoke-RunQueue -Exe ttr-train-ppo -Out "runs\ppo\pass4" -Common $common -Runs $runs -Seeds (0..2) -Slots 12 -Log pass4.log -DryRun:$DryRun
