# Second PPO pass (PLAN.md "Current progress", tier C). Pass 1 was still improving when its learning
# rate reached 0 at 50000 games, and shaping may not be needed with PPO. Pass 1's pool arm (same pool,
# default PPO settings, no ticket plan) for 100000 games, the learning rate annealed over all of them.
# Arms: with shaping 1 (pass 1, longer) and without shaping. 2 x 4 seeds, 8 at once, about 5 h.
#   powershell -ExecutionPolicy Bypass -File scripts\ppo_pass2.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/ppo/pass2/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/ppo/pass2/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# evaluations every 2000 games (100 games each vs
# random, greedy, wary, racer, collector and the held-out best linear agent p7b); best checkpoint on the
# mean over greedy, wary, racer, collector
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 100000 --opponent pool --pool $pool --eval-opponents random greedy wary racer collector --eval-every 2000 --live 2000"
$runs = [ordered]@{
  "p2a_pool_shape" = "--shaping 1"
  "p2b_pool"       = ""
}

Invoke-RunQueue -Exe ttr-train-ppo -Out "runs\ppo\pass2" -Common $common -Runs $runs -Seeds (0..3) -Slots 8 -Log pass2.log -DryRun:$DryRun
