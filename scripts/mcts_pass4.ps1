# Fourth MCTS pass (PLAN.md "Current progress", tier D). Pass 3: PPO's policy as the prior made the search race
# and beat PPO head to head (+7.5), but it fell 12.5 short of the ticket-playing search against greedy. Pass 3's
# "mcts:reward=score,guide=..." games are the paired control (same games: batch seed 9001).
#   reward=margin,guide=...           PPO was trained on margin, which takes points from greedy by tempo: does
#                                     margin close the guided search's gap to PPO against greedy? vs PPO, linear,
#                                     racer, greedy
#   guide=...,opponent=infer          the inferred opponent model under the guide: same opponents
#   guide=...,iterations=1600         4x the search: does the lead over PPO grow? vs PPO, 100 games, ~9 min a game
#   guide=... vs opponent=infer       the two strongest searches head to head, 100 games, ~4.5 min a game
# 12 processes, ~1000 games, ~4.5 h. Resumes where it stopped if rerun (rows already in the CSV are skipped).
#   powershell -ExecutionPolicy Bypass -File scripts\mcts_pass4.ps1        (-DryRun prints the commands)
# Watch:  .venv\Scripts\ttr-dash runs\mcts\pass4.csv runs\mcts\pass3.csv runs\mcts\pass1_reference.csv --control "mcts:reward=score,guide=ppo:runs/ppo/pass2/p2b_pool_s3.json" --live
param([switch]$DryRun)
Set-Location (Split-Path $PSScriptRoot)

$ppo = "ppo:runs/ppo/pass2/p2b_pool_s3.json"
$linear = "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"
$common = @("--workers", "12", "--out", "runs\mcts\pass4")
$batches = @(
  @("mcts:reward=margin,guide=$ppo", "mcts:reward=score,guide=$ppo,opponent=infer",
    "--opponents", $ppo, $linear, "racer", "greedy", "--games", "100", "--chunk", "2"),
  @("mcts:reward=score,guide=$ppo,iterations=1600", "--opponents", $ppo, "--games", "100", "--chunk", "1"),
  @("mcts:reward=score,guide=$ppo", "--opponents", "mcts:reward=score,opponent=infer", "--games", "100", "--chunk", "1")
)
. "$PSScriptRoot\run_queue.ps1"
if (-not $DryRun) { Set-KeepAwake $true }  # across batches: each eval_agents.py holds it only while it runs
try {
  foreach ($batch in $batches) {
    $cmd = @("scripts\eval_agents.py") + $batch + $common
    if ($DryRun) { ".venv\Scripts\python.exe $($cmd -join ' ')"; continue }
    & .venv\Scripts\python.exe @cmd
    if ($LASTEXITCODE -ne 0) { throw "eval_agents.py exited with $LASTEXITCODE" }
  }
} finally {
  if (-not $DryRun) { Set-KeepAwake $false }
}
