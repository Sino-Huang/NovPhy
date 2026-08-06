# Deterministic Unity 2019.4 staged builds

- Unity 2019.4.41f2 accepts `-deterministic` from `Assets/csc.rsp` and accepts `BuildOptions.NoUniqueIdentifier | BuildOptions.StrictMode`.
- Roslyn determinism stabilizes managed PE checksum/MVID bytes; `NoUniqueIdentifier` stabilizes Unity's generated player build identity.
- With the stable Python tar/gzip packager, two fresh exact-SHA builds produced identical archive, provenance, and receipt bytes.
- Keep the response file's Unity `.meta` committed so project import does not create untracked metadata.
