# Third PPO pass (PLAN.md "Current progress", tier C): the reward mode, an open experiment since the
# design ("Reward"): does racing (and would blocking) only come from a reward that includes the
# opponent? margin = my score - the opponent's (every agent so far); score = my own points only
# (nothing gained by hurting the opponent or ending early); win = +1 / -1 at the end only (a safe
# +5 counts as much as a +40: risk tolerance). PPO pass 2's no-shaping setting (same pool, default
# PPO settings) at pass 1's length, 50000 games (plateau from ~30-50k at that learning-rate decay).
# 3 arms x 4 seeds, all 12 at once, about 4-5 h.
#   powershell -ExecutionPolicy Bypass -File scripts\ppo_pass3.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/ppo/pass3/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/ppo/pass3/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# evaluations every 2000 games (100 games each vs random, greedy, wary, racer, collector and the
# held-out best linear agent p7b), reported as score margins whatever the training reward; best
# checkpoint on the mean margin over greedy, wary, racer, collector
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 50000 --opponent pool --pool $pool --eval-opponents random greedy wary racer collector --eval-every 2000 --live 2000"
$runs = [ordered]@{
  "p3a_margin" = "--reward margin"
  "p3b_score"  = "--reward score"
  "p3c_win"    = "--reward win"
}

Invoke-RunQueue -Exe ttr-train-ppo -Out "runs\ppo\pass3" -Common $common -Runs $runs -Seeds (0..3) -Slots 12 -Log pass3.log -DryRun:$DryRun
