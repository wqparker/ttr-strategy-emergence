# First MCTS pass (PLAN.md "Current progress", tier D): no training, so a pass is an evaluation. The default
# search (400 iterations, PUCT with greedy's move as the prior, only main actions and ticket choices searched,
# greedy rollouts and opponent model) under the margin and own-score rewards, against every scripted bot, the
# reference strongest agent (PPO p2b_pool_s3) and the strongest linear agent (p7b). 100 games per pair on the
# re-score games (batch seed 9001), 12 processes, about 3 min a game: ~5 h. Resumes where it stopped if rerun.
#   powershell -ExecutionPolicy Bypass -File scripts\mcts_pass1.ps1        (-DryRun prints the command)
# Watch:  Get-Content runs\mcts\pass1.log -Wait
param([switch]$DryRun)
Set-Location (Split-Path $PSScriptRoot)

$agents = @("mcts", "mcts:reward=score")
$opponents = @("greedy", "racer", "wary", "collector", "ppo:runs/ppo/pass2/p2b_pool_s3.json",
               "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best")
$cmd = @("scripts\eval_agents.py") + $agents + @("--opponents") + $opponents +
        @("--games", "100", "--chunk", "4", "--workers", "12", "--out", "runs\mcts\pass1")
if ($DryRun) { ".venv\Scripts\python.exe $($cmd -join ' ')"; return }
& .venv\Scripts\python.exe @cmd
