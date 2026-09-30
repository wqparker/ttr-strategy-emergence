# First PPO pass (PLAN.md "Current progress", tier C): DQN pass 3's setting with PPO, 50000 games (PPO
# needs more samples: at 3000 games it was at -70 to -84 vs greedy where DQN was near -45). The pool:
# greedy, wary, racer twice, collector, self, and two strong linear agents; shaping 1; default PPO settings.
# Arms: without and with the observation's ticket-plan block. 2 x 4 seeds, 50000 games, 8 at once.
#   powershell -ExecutionPolicy Bypass -File scripts\ppo_pass1.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/ppo/pass1/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/ppo/pass1/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# evaluations every 1000 games (100 games each vs
# random, greedy, wary, racer, collector and the held-out best linear agent p7b); best checkpoint on the
# mean over greedy, wary, racer, collector
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 50000 --opponent pool --pool $pool --shaping 1 --eval-opponents random greedy wary racer collector --eval-every 1000 --live 1000"
$runs = [ordered]@{
  "p1a_pool"      = ""
  "p1b_pool_plan" = "--ticket-plan"
}

Invoke-RunQueue -Exe ttr-train-ppo -Out "runs\ppo\pass1" -Common $common -Runs $runs -Seeds (0..3) -Slots 8 -Log pass1.log -DryRun:$DryRun
