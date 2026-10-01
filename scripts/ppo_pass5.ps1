# Fifth PPO pass (PLAN.md "Current progress", tier C): the two multiplayer settings left after pass 4, where
# racing survived at 3-4 players against the pool and nobody blocked. 5 players: the nine 6-routes shared
# five ways, both sides of double routes open. 4 players in self-play only (--pool self): every opponent seat
# a frozen copy of the learner (the best network or one of the 5 latest evaluated; greedy until the first
# evaluation), which removes the pool's racing bias (5 of its 8 members race). Margin and score rewards each,
# as in pass 4 (only margin rewards blocking). Otherwise pass 4's setting (no shaping, default PPO settings),
# 50000 games, 3 seeds; evaluated against the bots as in pass 4, so all arms compare. All 12 at once.
#   powershell -ExecutionPolicy Bypass -File scripts\ppo_pass5.ps1        (-DryRun prints the commands)
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/ppo/pass5/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/ppo/pass5/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# evaluations every 2000 games (100 games each vs random, greedy, wary, racer, collector and the held-out
# best linear agent p7b, each filling every other seat), reported as score margins against the opponents'
# mean whatever the training reward; best checkpoint on the mean margin over greedy, wary, racer, collector
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 50000 --opponent pool --eval-opponents random greedy wary racer collector --eval-every 2000 --live 2000"
$runs = [ordered]@{
  "p5a_5p_margin"      = "--players 5 --pool $pool --reward margin"
  "p5b_5p_score"       = "--players 5 --pool $pool --reward score"
  "p5c_4p_self_margin" = "--players 4 --pool self --reward margin"
  "p5d_4p_self_score"  = "--players 4 --pool self --reward score"
}

Invoke-RunQueue -Exe ttr-train-ppo -Out "runs\ppo\pass5" -Common $common -Runs $runs -Seeds (0..2) -Slots 12 -Log pass5.log -DryRun:$DryRun
