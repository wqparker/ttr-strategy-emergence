# Second MCTS pass (PLAN.md "Current progress", tier D). Pass 1's own-score search beat every scripted bot,
# racer by more than PPO, but lost to PPO head to head: it never ended those games and failed tickets, as if
# its greedy opponent model expected a longer game than PPO plays. All arms use own score; pass 1's
# "mcts:reward=score" games are the paired control (same games: batch seed 9001).
#   opponent=racer         the tempo forecast fixed by hand: vs PPO, linear, racer, greedy
#   opponent=greedy+racer  a mixed opponent model, each sampled world one or the other: same opponents
#   prior=0.25             a weaker pull toward greedy's move: vs PPO, racer, greedy
#   iterations=1600        4x the search: vs PPO and racer, 50 games (about 9 min a game)
# 12 processes, ~1250 games, ~5-6 h. Resumes where it stopped if rerun (rows already in the CSV are skipped).
#   powershell -ExecutionPolicy Bypass -File scripts\mcts_pass2.ps1        (-DryRun prints the commands)
# Watch:  .venv\Scripts\python scripts\eval_agents.py --report 30 --out runs\mcts\pass2
param([switch]$DryRun)
Set-Location (Split-Path $PSScriptRoot)

$ppo = "ppo:runs/ppo/pass2/p2b_pool_s3.json"
$linear = "linear:runs/linear/pass7/p7b_sarsa_lam98_random_s3.json@best"
$common = @("--workers", "12", "--out", "runs\mcts\pass2")
$batches = @(
  @("mcts:reward=score,opponent=racer", "mcts:reward=score,opponent=greedy+racer",
    "--opponents", $ppo, $linear, "racer", "greedy", "--games", "100", "--chunk", "2"),
  @("mcts:reward=score,prior=0.25", "--opponents", $ppo, "racer", "greedy", "--games", "100", "--chunk", "2"),
  @("mcts:reward=score,iterations=1600", "--opponents", $ppo, "racer", "--games", "50", "--chunk", "1")
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
