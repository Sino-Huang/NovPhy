# Unity 2019.4.41f2 SIGSEGV Audit Hypotheses

## H1: CEF browser-loop incompatibility

Claim: CEF causes a SIGSEGV during headless Unity startup.

Distinguishing evidence: a probe must start the exact Editor and emit a CEF-related crash or SIGSEGV signature. The three probe logs instead show the dynamic loader failing on `libgconf-2.so.4`; probes 2 and 3 captured exit `127` and no signal, while probe 1's wrapper failed before appending result metadata.

Status: Not evaluated. No CEF execution was observed.

## H2: Project/import state

Claim: migration-project state triggers the crash.

Distinguishing evidence: the exact Editor must be compared between a non-project launch and a fresh empty project. Both were attempted as non-project/fresh-project probes, but neither passed the loader stage.

Status: Not evaluated. No project initialization was observed, and the migration project was not run.

## H3: Graphics/display/library environment

Claim: graphics, display, or library-path conditions trigger the crash.

Distinguishing evidence: the no-project baseline and `-disable-gpu -force-gfx-st` variant must reach different runtime stages or show different crash signatures. Both stopped identically at the missing `libgconf-2.so.4` dependency.

Status: External launch blocker observed; graphics/display behavior was not evaluated.

## Audit conclusion

All three required variants were attempted with the exact `2019.4.41f2` Editor and the same private `LD_LIBRARY_PATH` value. All three failed before Unity startup with the same missing system dependency. This is an external environment blocker, not evidence of CEF SIGSEGV and not evidence of EditMode success.
