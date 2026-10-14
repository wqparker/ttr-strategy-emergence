# Expert iteration, pass 1, stage 3 (PLAN.md "Current progress"): PPO from the distilled ticket network (stage 2,
# scripts\exit_pass1_distill.ps1). The network starts out playing tickets; does reinforcement learning under own
# score keep them, improve on them, or drift back to the race every learner found from scratch? Each arm is a PPO
# pass 6 / pass 3 setting with only the starting network changed (own score, default PPO settings, 50000 games):
#   f1a_greedy       vs greedy only                          (pass 6 p6a: raced from scratch, 92 own points)
#   f1b_ticket_pool  vs greedy, wary, collector              (pass 6 p6d)
#   f1c_pool         pass 3's pool, racers and self included (pass 3 p3b_score)
#   f1d_greedy_slow  f1a at a fifth of the learning rate: is any drift only the step size?
# 4 arms x 3 seeds, all 12 at once, about 2.5-3 h.
#   powershell -ExecutionPolicy Bypass -File scripts\exit_pass1_ppo.ps1 -Init runs\exit\pass1\NAME.json@best [-TicketPlan]
# Watch:  .venv\Scripts\python scripts\run_status.py "runs/exit/pass1/f1*.json"
param([Parameter(Mandatory)][string]$Init, [switch]$TicketPlan, [switch]$DryRun)
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# as pass 6: evaluations every 2000 games (100 games each vs random, greedy, wary, racer, collector and the held-out
# linear agent p7b), score margins whatever the training reward; best checkpoint on the mean over all but random
$pool = "greedy wary racer racer collector self linear:runs/linear/pass7/p7d_q_lam98_self_s2.json@best linear:runs/linear/pass8/p8b_q_lam98_self_decay_s4.json@best"
$common = "--games 50000 --reward score --init $Init$(if ($TicketPlan) { ' --ticket-plan' }) " +
          "--eval-opponents random greedy wary racer collector --eval-every 2000 --live 2000"
$runs = [ordered]@{
  "f1a_greedy"      = "--opponent greedy"
  "f1b_ticket_pool" = "--opponent pool --pool greedy wary collector"
  "f1c_pool"        = "--opponent pool --pool $pool"
  "f1d_greedy_slow" = "--opponent greedy --lr 5e-5"
}

Invoke-RunQueue -Exe ttr-train-ppo -Out "runs\exit\pass1" -Common $common -Runs $runs -Seeds (0..2) -Slots 12 -Log ppo.log -DryRun:$DryRun
