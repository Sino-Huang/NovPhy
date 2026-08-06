# Todo 3 Physics Recorder QA

- `physics_capture_v1` contacts are canonicalized by lexical lifetime `entity_id`, not collider ID; swapping participants negates both normal and relative velocity.
- `support_v1` requires the same canonical pair in fixed steps `n-1` and `n`, validates the support predicates in both samples, and retains both contact IDs and steps.
- Recorder record and byte limits apply to raw contacts, supports, and events. Death and destroy normalize to one `entity_destroyed` event per lifetime.
- Unity 2019.4.41f2 in this environment needs a temporary extracted Debian `libgconf-2.so.4` on `LD_LIBRARY_PATH`; the no-`-quit` bounded invocation allows the async Test Framework to write XML, while `-quit` may exit before refreshing an existing result file.
- Required focused receipts: `task-3-physics.xml` and `task-3-physics-repeat.xml`, each `7/7` passed. The unfiltered full suite remains `bounded_no_xml` and is not a pass claim.
