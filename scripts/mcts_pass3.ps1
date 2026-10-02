# Third MCTS pass (PLAN.md "Current progress", tier D). Pass 2: racer as the opponent model helped against the
# racing agents (PPO, linear) and cost against greedy; 4x the search also reached parity with PPO. All arms use
# own score; pass 1's "mcts:reward=score" games are the paired control (same games: batch seed 9001).
#   opponent=infer                  the opponent model chosen from its public play (racer_belief): racer's
#                                   gains against racers without its cost against ticket players. Every opponent
#   guide=ppo:...p2b_pool_s3        PPO's policy as the PUCT prior in place of greedy's move (greedy rollouts
#                                   still judge the moves): vs PPO, linear, racer, greedy
#   opponent=racer,iterations=1600  pass 2's two gains together: can it pass PPO? 200 games, ~9.5 min a game
# 12 processes, ~1200 games, ~6-7 h. Resumes where it stopped if rerun (rows already in the CSV are skipped).
#   powershell -ExecutionPolicy Bypass -File scripts\mcts_pass3.ps1        (-DryRun prints the commands)
# Watch:  .venv\Scripts\ttr-dash runs\mcts\pass3.csv runs\mcts\pass1.csv runs\mcts\pass1_reference.csv --control "mcts:reward=score" --live
param([switch]$DryRun)
Set-Location (Split-Path $PSScriptRoot)

$ppo = "ppo:runs/ppo/pass2/p2b_pool_s3.json"
$linear = "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"
$common = @("--workers", "12", "--out", "runs\mcts\pass3")
$batches = @(
  @("mcts:reward=score,opponent=infer",
    "--opponents", $ppo, $linear, "racer", "greedy", "wary", "collector", "--games", "100", "--chunk", "2"),
  @("mcts:reward=score,guide=$ppo", "--opponents", $ppo, $linear, "racer", "greedy", "--games", "100", "--chunk", "2"),
  @("mcts:reward=score,opponent=racer,iterations=1600", "--opponents", $ppo, "--games", "200", "--chunk", "1")
)
foreach ($batch in $batches) {
  $cmd = @("scripts\eval_agents.py") + $batch + $common
  if ($DryRun) { ".venv\Scripts\python.exe $($cmd -join ' ')"; continue }
  & .venv\Scripts\python.exe @cmd
  if ($LASTEXITCODE -ne 0) { throw "eval_agents.py exited with $LASTEXITCODE" }
}
