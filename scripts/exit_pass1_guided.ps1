# Expert iteration, pass 1, stage 4 (PLAN.md "Current progress"): the search guided by a ticket network, ExIt's next
# expert. The teacher searched with greedy's move as the PUCT prior; here the prior is a network's policy, with the
# teacher's rollouts and inferred opponent model:
#   x1c   the distilled network (x1c_plan_s1@best: visit-count targets, so its probabilities have the search's
#         spread; ticket plan)
#   f1    the distilled network after PPO (-Guide2, a stage 3 run: ticket play kept and improved)
# Is either guided search stronger than the teacher? If so, its games are the next round's training data.
# 100 games against each of PPO p2b, linear p7b, racer, greedy on the re-score games (batch seed 9001), paired with
# pass 3's "mcts:reward=score,opponent=infer" (the teacher) on the same games. 12 processes, about 3 h.
#   powershell -ExecutionPolicy Bypass -File scripts\exit_pass1_guided.ps1 -Guide2 runs/exit/pass1/NAME.json  (-DryRun)
# Watch:  .venv\Scripts\python scripts\eval_agents.py --out runs\exit\pass1_guided --report 60
param([Parameter(Mandatory)][string]$Guide2, [switch]$DryRun)
Set-Location (Split-Path $PSScriptRoot)

$guided = @("mcts:reward=score,opponent=infer,guide=ppo:runs/exit/pass1/x1c_plan_s1.json@best",
            "mcts:reward=score,opponent=infer,guide=ppo:$Guide2")
$ppo = "ppo:runs/ppo/pass2/p2b_pool_s3.json"
$linear = "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"
$cmd = @("scripts\eval_agents.py") + $guided + @("--opponents", $ppo, $linear, "racer", "greedy",
         "--games", "100", "--chunk", "2", "--workers", "12", "--out", "runs\exit\pass1_guided")
if ($DryRun) { ".venv\Scripts\python.exe $($cmd -join ' ')"; return }
. "$PSScriptRoot\run_queue.ps1"
Set-KeepAwake $true
try {
  & .venv\Scripts\python.exe @cmd
  if ($LASTEXITCODE -ne 0) { throw "eval_agents.py exited with $LASTEXITCODE" }
} finally {
  Set-KeepAwake $false
}
