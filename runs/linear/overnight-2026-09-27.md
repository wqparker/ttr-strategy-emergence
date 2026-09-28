# Overnight training log, 2026-09-27

Claude's record of what ran and every action taken. Plan agreed: fewer, longer passes;
existing `ttr-train-linear` options only, no learner code changes; stop only runs that are
broken (crash, non-finite weights) or collapsed (past 3000 games, below -150 vs random with
under 2 claims a game). A setting that is merely weak is a result, not a failure.

- 01:17 Pass 4 started (`scripts/linear_pass4.ps1`): 6 settings x 3 seeds x 10000 games.
- 01:25 Keep-awake process started (PID 22680, 9 h), so the PC doesn't sleep between passes.
- 01:25 Early read at ~1500 games: all healthy. Shaping without lambda leads (-9 vs greedy,
  37% wins, 1.7 tickets completed); lambda 0.98 at alpha 0.2 no longer collapses.
- 01:35 CPU (i9-11900K) at ~85 C under full load, per the user: fine. Pass 5 runs a few
  fewer processes than pass 4's 18 to go easier on it.
- 01:37 Watcher alarm "10 processes for 11 unfinished runs": false alarm, a run finished
  between the file read and the process count. Watcher now needs it on two polls in a row.
- 01:41 Pass 4 finished, all 18 exit 0 (24 min, far faster than estimated). Results and
  readings in PLAN.md. Short version: every setting -20 to -5 vs greedy (pass 3 was -57 to
  -164); averaging is the big gain; the lambda collapse was a too-small step; the fast
  settings plateau near -5 (parity with greedy) when trained against random.
- 01:43 Pass 5 started (`scripts/linear_pass5.ps1`): p4b's setting, 30000 games, 5 seeds,
  at most 12 at a time (25 runs). Q vs random / SARSA vs random / Q vs greedy / Q vs mixed /
  Q vs self. Smoke-tested SARSA and self-play with averaging first.
- 02:14 Check: first wave (12 runs) done in ~27 min, second wave running, all healthy.
  p5e_q_self_s1 drifted to -75 vs greedy by game 29000 (best +2 at game 4000): self-play
  instability, a result, left running.
- 02:44 Pass 5: 24 of 25 done, all healthy; p5e_q_self_s4 (queued last) still running
  alone. Results in PLAN.md. Short version: no training opponent gets past parity; every
  Q run vs a real opponent peaks near game 5000 then declines at constant alpha (self-play
  collapses to -58 vs greedy); SARSA is steadier and still rising; mixed makes every seed
  a ticket dumper.
- 02:50 Pass 6 started (`scripts/linear_pass6.ps1`): alpha decay 0.05 -> 0.005 over
  {Q, SARSA} x {random, greedy, self}, plus constant-alpha SARSA vs greedy and vs self.
  40 runs, 12 at a time, est. ~1.5-2 h. Smoke-tested alpha decay with SARSA self-play first.
- 03:04 Pass 5 complete, all 25 exit 0. p5e_q_self_s4 ended at -30; self-play row now -58 over 5 seeds.
- 03:16 Pass 6 check: 10 of 40 done, healthy. Early: alpha decay helps Q vs greedy (~-6); self-play
  collapses vs greedy with SARSA and with alpha decay too, so it is not a Q-learning instability.
- 03:46, 04:16 Pass 6 checks: healthy, no error output.
- 04:18 Pass 6 at 36/40: analysed the finished runs. Alpha decay does not stop the late
  decline, and SARSA declines vs greedy too, so it isn't Q-specific divergence. Weights stay
  small (|w| ~1); the policy drifts from ~12 claims of length ~3.7 ending the game itself to
  ~20 claims of length 2 that rarely end it. Training margin falls with evaluation margin.
- 04:18 Wrote `scripts/rescore.py` (plays final and best weights vs greedy on the same 1000
  fresh games, seed 9001) and re-scored passes 4-5 in 7 min: `runs/linear/rescore_pass4-5.*`.
  Best checkpoints genuinely beat greedy: Q vs greedy +3.3 (all 5 seeds positive, 54% wins),
  Q vs self +3.0, p4f (lambda 0.98, alpha 0.2) +4.7, and p4f_lam98_s1@best +13.4 +- 1.0
  (66% wins), the strongest linear agent so far. Finals: -5 to -52.
- 04:27 Pass 7 started (`scripts/linear_pass7.ps1`): p4f's setting (lambda 0.98, alpha 0.2,
  average 100), 30000 games, 5 seeds: Q vs random / SARSA vs random / Q vs greedy / Q vs self.
  20 runs, 12 at a time. Smoke-tested SARSA(lambda) and lambda self-play first.
- 04:35 Pass 6 complete, all 40 exit 0. Results in PLAN.md.
- 04:49 Re-scored pass 6 (`runs/linear/rescore_pass6.*`). Most reliable greedy-beater so far:
  Q vs greedy with alpha decay, best checkpoint +7.4 (every seed +6 to +9, 59% wins). Q beats
  SARSA vs greedy and self. PLAN.md has the full re-score table for passes 4-6.
- 04:57 Pass 7 check: healthy. (Watcher re-armed.)
- 05:25 Pass 7 complete, all 20 exit 0 (the second wave of 8 ran alone, ~25 min).
- 05:25-09:23 **PC asleep.** It slept the moment pass 7's launcher released its keep-awake
  request. The separate 9-hour keep-awake process (PID 22680) never held one: its command was
  passed inline through Start-Process, which dropped the inner quotes, so the P/Invoke never
  compiled and it only ran Start-Sleep. The launcher scripts (run from files) worked, so the
  PC stayed awake whenever a pass was running. Woken 09:23 by the network adapter. No training
  lost: every pass had finished. Next time: put the keep-awake in a script file too.
- 09:28 (thought it was ~05:30 when deciding) Re-scored pass 7 and started pass 8
  (`scripts/linear_pass8.ps1`): lambda 0.98 + alpha decay 0.2 -> 0.02, Q vs greedy and vs self,
  5 seeds, 10 runs at once, ~35 min. Stopped the ineffective keep-awake process.
- ~10:08 Pass 8 launcher died before finishing (no exit or finished lines in `pass8.log`; cause
  unverified, likely its console closing). p8a all 5 and p8b s0 finished; p8b s1-s4 stopped at
  27000-28000 games with their stdout logs empty (buffered output lost); their JSON live saves
  are intact.
- 23:45 Re-scored pass 8 (`runs/linear/rescore_pass8.*`). alpha decay makes lambda-0.98 finals
  usable vs greedy (p8a final +5.6, four seeds +8 to +13) but best checkpoints stay at +9 to +10,
  as in pass 7. Results in PLAN.md.
