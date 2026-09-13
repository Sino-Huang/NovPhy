# Separate post-fit validation allowance

The fit source/plan frozen at `5bcfac7` remain unchanged while the six
continuations execute. After their completed report exists, run
`python -m scripts.validate_issue_76_lower_rate_dynamics` in initialized
novphy with env.sh sourced. Use a separately recorded cumulative180 CPU-second
validation allowance with the existing12GiB RSS limit; no fitting or GPU work.

Recompute all30 assigned rows and180 curves exactly, including all preserved
references and qualification decisions. Check all six completed counters,
every parameter's optimizer-step increment against the actual source state,
and all optimizer group settings against source settings with only the
declared learning-rate change. Verify the complete12-job stage cost inventory
and resource limits. Keep validation cost separate from the original report
and publish its finished snapshot with validation. A failed training
qualification is valid negative evidence, not a validation error or gate pass.
