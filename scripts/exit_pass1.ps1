# Expert iteration, pass 1 (PLAN.md "Current progress"): can a network hold a ticket strategy? Every learner races
# and abandons its 2 opening tickets, even trained against ticket players only (PPO pass 6); the own-score search
# plays tickets (keeps 5.5 and completes 5.4 against greedy) and is the strongest agent against the ticket players.
# Stage 1, this script: the search's moves as training data. The teacher is pass 3's mcts:reward=score,opponent=infer
# (400 iterations, greedy's move as the prior, the opponent's style inferred), 500 games against each of greedy,
# wary, collector, racer, PPO p2b and linear p7b on batch seed 7001 (fresh: the re-score games are 9001). Every
# decision with a choice is recorded (observation with the ticket-plan block, mask, move, root visit counts and
# values, the game's result): runs\exit\pass1\data\*.npz; the teacher's games in runs\exit\pass1\teacher.csv.
# 12 processes, 3000 games, about 2.3 min a game: ~9.5 h. Resumes where it stopped if rerun.
# Stage 2 (ttr-train-distill on the data) and stage 3 (PPO from the distilled network) follow once it is done.
#   powershell -ExecutionPolicy Bypass -File scripts\exit_pass1.ps1        (-DryRun prints the command)
# Watch:  .venv\Scripts\python scripts\eval_agents.py --out runs\exit\pass1\teacher --report 60
param([switch]$DryRun)
Set-Location (Split-Path $PSScriptRoot)

$teacher = "mcts:reward=score,opponent=infer"
$ppo = "ppo:runs/ppo/pass2/p2b_pool_s3.json"
$linear = "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"
$cmd = @("scripts\eval_agents.py", $teacher, "--opponents", "greedy", "wary", "collector", "racer", $ppo, $linear,
         "--games", "500", "--chunk", "4", "--seed", "7001", "--workers", "12",
         "--data", "runs\exit\pass1\data", "--out", "runs\exit\pass1\teacher")
if ($DryRun) { ".venv\Scripts\python.exe $($cmd -join ' ')"; return }
. "$PSScriptRoot\run_queue.ps1"
Set-KeepAwake $true
try {
  & .venv\Scripts\python.exe @cmd
  if ($LASTEXITCODE -ne 0) { throw "eval_agents.py exited with $LASTEXITCODE" }
} finally {
  Set-KeepAwake $false
}
