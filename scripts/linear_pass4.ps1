# Fourth linear pass (PLAN.md "Current progress"): 6 settings x 3 seeds, 10000 games each,
# all in parallel. Keeps the machine awake until every run ends, then lets it sleep again.
#   powershell -ExecutionPolicy Bypass -File scripts\linear_pass4.ps1
# Watch:  .venv\Scripts\ttr-dash.exe --live --group "runs/linear/pass4/*.json"
param([switch]$DryRun)  # -DryRun: print the command lines, start nothing
Set-Location (Split-Path $PSScriptRoot)
. "$PSScriptRoot\run_queue.ps1"

# epsilon reaches 0.02 at game 1000, as in passes 2 and 3, so their first 2000 games compare
$common = "--algo q --opponent random --games 10000 --epsilon-decay 0.1 --live 100"
$runs = [ordered]@{
  "p4a_base"  = "--alpha 0.05"                                  # pass 2's q_random, run longer
  "p4b_avg"   = "--alpha 0.05 --average 100"                    # averaging at pass 2's alpha
  "p4c_slow"  = "--alpha 0.02 --average 100"                    # p3a run longer: slower or worse?
  "p4d_shape" = "--alpha 0.05 --average 100 --shaping 1"        # shaping without lambda
  "p4e_lam90" = "--alpha 0.05 --average 100 --lambda 0.9"       # a shorter trace
  "p4f_lam98" = "--alpha 0.2 --average 100 --lambda 0.98"       # lambda 0.98 with a larger alpha
}

Invoke-RunQueue -Exe ttr-train-linear -Out "runs\linear\pass4" -Common $common -Runs $runs -Seeds (0..2) -Slots 18 -Log pass4.log -DryRun:$DryRun
