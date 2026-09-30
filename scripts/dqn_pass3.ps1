# Third DQN pass (PLAN.md "Current progress", tier B): a stronger pool and ticket features. The
# pool: greedy, wary, racer (fixed: it falls back to shorter routes once the 6-routes are gone; pass 2
# exploited that) twice, collector, self, and two strong linear agents. Shaping 1, n = 1, averaging 100.
# Arms: without and with the observation's ticket-plan block. 2 x 4 seeds, 30000 games, 8 at once.
#   powershell -ExecutionPolicy Bypass -File scripts\dqn_pass3.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/dqn/pass3/*.json"
#         .venv\Scripts\python scripts\run_status.py "runs/dqn/pass3/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon 1.0 -> 0.02 over the first 1500 games; evaluations every 1000 games (100 games each vs
# random, greedy, wary, racer, collector and the held-out best linear agent p7b); best checkpoint on the
# mean over greedy, wary, racer, collector
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 30000 --n-step 1 --average 100 --opponent pool --pool $pool --shaping 1 --eval-opponents random greedy wary racer collector --eval-every 1000 --live 1000"
$runs = [ordered]@{
  "d3a_pool"      = ""
  "d3b_pool_plan" = "--ticket-plan"
}

Invoke-RunQueue -Exe ttr-train-dqn -Out "runs\dqn\pass3" -Common $common -Runs $runs -Seeds (0..3) -Slots 8 -Log pass3.log -DryRun:$DryRun
