# Sixth PPO pass (PLAN.md "Current progress", tier C): can a learner find tickets when its opponents don't race?
# MCTS passes 1-4: under own score a search wins by tickets against greedy (keeps 5.5, completes 5.4, scores 136),
# so tickets beat racing there, but no learner has ever found them: every pool had racers in it (5 of 8 seats).
# Own score throughout (nothing gained by ending the game early or hurting the opponent), opponents that play
# tickets only, 2 players, otherwise pass 3's setting (no shaping, default PPO settings, 50000 games).
#   p6a  vs greedy only                     separates "racing is the best response to the pool" from "tickets
#                                           are too hard to learn"
#   p6b  + ticket-plan observation          exact ticket computations (DQN: doubled tickets completed)
#   p6c  + shaping 1                        credit for progress toward tickets before the game ends
#   p6d  pool of ticket players             greedy, wary, collector: not tuned to one opponent
# 4 arms x 3 seeds, all 12 at once, about 2.5-3 h.
#   powershell -ExecutionPolicy Bypass -File scripts\ppo_pass6.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/ppo/pass6/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/ppo/pass6/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# evaluations every 2000 games (100 games each vs random, greedy, wary, racer, collector and the
# held-out best linear agent p7b), reported as score margins whatever the training reward; best
# checkpoint on the mean margin over greedy, wary, racer, collector
$common = "--games 50000 --reward score --eval-opponents random greedy wary racer collector --eval-every 2000 --live 2000"
$runs = [ordered]@{
  "p6a_greedy"        = "--opponent greedy"
  "p6b_greedy_plan"   = "--opponent greedy --ticket-plan"
  "p6c_greedy_shape"  = "--opponent greedy --shaping 1"
  "p6d_ticket_pool"   = "--opponent pool --pool greedy wary collector"
}

Invoke-RunQueue -Exe ttr-train-ppo -Out "runs\ppo\pass6" -Common $common -Runs $runs -Seeds (0..2) -Slots 12 -Log pass6.log -DryRun:$DryRun
