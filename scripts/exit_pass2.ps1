# Expert iteration, pass 2 (PLAN.md "Current progress"): the second round. Pass 1's best expert is the teacher:
# the search guided by the PPO-improved ticket network f1d_greedy_slow_s2 (stage 4: +8.4 a game over pass 1's
# teacher across PPO p2b, linear p7b, racer and greedy; +5.9 against PPO; tickets against every opponent).
# Stage 1, this script: its moves as training data, as pass 1: 500 games against each of greedy, wary, collector,
# racer, PPO p2b and linear p7b, on batch seed 7002 (fresh: pass 1 recorded 7001, re-scores use 9001). Every decision
# with a choice is recorded: runs\exit\pass2\data\*.npz; the teacher's games in runs\exit\pass2\teacher.csv.
# 12 processes, 3000 games, about 2.2 min a game: ~9 h. Resumes where it stopped if rerun.
# Stage 2: scripts\exit_pass2_distill.ps1 (starts by itself after this with -WaitPid).
#   powershell -ExecutionPolicy Bypass -File scripts\exit_pass2.ps1        (-DryRun prints the command)
# Watch:  .venv\Scripts\python scripts\eval_agents.py --out runs\exit\pass2\teacher --report 60
param([switch]$DryRun)
Set-Location (Split-Path $PSScriptRoot)

$teacher = "mcts:reward=score,opponent=infer,guide=ppo:runs/exit/pass1/f1d_greedy_slow_s2.json"
$ppo = "ppo:runs/ppo/pass2/p2b_pool_s3.json"
$linear = "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"
$cmd = @("scripts\eval_agents.py", $teacher, "--opponents", "greedy", "wary", "collector", "racer", $ppo, $linear,
         "--games", "500", "--chunk", "4", "--seed", "7002", "--workers", "12",
         "--data", "runs\exit\pass2\data", "--out", "runs\exit\pass2\teacher")
if ($DryRun) { ".venv\Scripts\python.exe $($cmd -join ' ')"; return }
. "$PSScriptRoot\run_queue.ps1"
Set-KeepAwake $true
try {
  & .venv\Scripts\python.exe @cmd
  if ($LASTEXITCODE -ne 0) { throw "eval_agents.py exited with $LASTEXITCODE" }
} finally {
  Set-KeepAwake $false
}
