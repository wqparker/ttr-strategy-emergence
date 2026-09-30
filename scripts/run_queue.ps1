# The run queue the training launchers share (scripts\*_pass*.ps1). A launcher dot-sources this file
# and calls, for example:
#   Invoke-RunQueue -Exe ttr-train-ppo -Out runs\ppo\pass4 -Common $common -Runs $runs -Seeds (0..2) -Slots 12
# One run per setting and seed, seed by seed: .venv\Scripts\EXE.exe "COMMON RUNS[name] --seed S --out
# OUT\name_sS.json", its output in OUT\name_sS.log and .err, at most SLOTS at a time. The PC stays awake
# until every run has ended. OUT\LOG gets a line when the queue starts, when each run starts, each run's
# exit code, and when it finishes. -DryRun prints the command lines and starts nothing.
function Invoke-RunQueue {
  param(
    [Parameter(Mandatory)][string] $Exe,
    [Parameter(Mandatory)][string] $Out,
    [Parameter(Mandatory)][string] $Common,
    [Parameter(Mandatory)][System.Collections.IDictionary] $Runs,
    [int[]] $Seeds = @(0),
    [int] $Slots = 12,
    [string] $Log = "queue.log",
    [switch] $DryRun
  )
  $queue = [System.Collections.Queue]::new()
  foreach ($s in $Seeds) { foreach ($name in $Runs.Keys) { $queue.Enqueue(@($name, $s)) } }
  if ($DryRun) {
    foreach ($item in $queue) {
      $name, $s = $item
      "$Exe $Common $($Runs[$name]) --seed $s --out $Out\${name}_s$s.json"
    }
    return
  }

  New-Item -ItemType Directory -Force $Out | Out-Null
  $logFile = Join-Path $Out $Log
  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while the queue runs
  if (-not ("Win32.Power" -as [type])) {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition `
      '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
  }
  [Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
  try {
    "started $(Get-Date -Format s)" | Out-File -Encoding utf8 $logFile
    $jobs = @()
    while ($queue.Count -gt 0 -or ($jobs.Proc | Where-Object { -not $_.HasExited })) {
      while ($queue.Count -gt 0 -and @($jobs.Proc | Where-Object { -not $_.HasExited }).Count -lt $Slots) {
        $name, $s = $queue.Dequeue()
        $run = "${name}_s$s"
        $proc = Start-Process -NoNewWindow -PassThru -FilePath ".venv\Scripts\$Exe.exe" `
          -ArgumentList "$Common $($Runs[$name]) --seed $s --out $Out\$run.json" `
          -RedirectStandardOutput "$Out\$run.log" -RedirectStandardError "$Out\$run.err"
        $null = $proc.Handle  # keep the handle, or ExitCode reads empty after exit
        $jobs += [pscustomobject]@{ Run = $run; Proc = $proc }
        "start $(Get-Date -Format s)  $run" | Out-File -Append -Encoding utf8 $logFile
      }
      Start-Sleep -Seconds 10
    }
    foreach ($j in $jobs) { "exit $($j.Proc.ExitCode)  $($j.Run)" | Out-File -Append -Encoding utf8 $logFile }
    "finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $logFile
  } finally {
    [Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null  # let the PC sleep again
  }
}
