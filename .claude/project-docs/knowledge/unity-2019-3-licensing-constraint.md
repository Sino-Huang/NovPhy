# Unity 2019.3 licensing constraint

As of 2026-08-01, Unity Personal uses Named User Licensing and does not provide serial keys or manual `.alf`/`.ulf` activation. Unity's current compatibility guidance only supports the 2019 stream through Named User Licensing from Unity 2019.4.27 onward. Consequently, the project's pinned Unity 2019.3.4f1 editor cannot be activated through a current Personal entitlement.

The supported exact-version route is a legitimate serial-based Pro, Enterprise, Industry, Student, or Educator entitlement: generate a machine-specific `.alf` with the 2019.3.4f1 editor, upload it at `https://license.unity3d.com/manual`, enter the serial, download the `.ulf`, and import it with `-manualLicenseFile`.

Do not pass Unity credentials or serials in shell commands, logs, issue comments, or chat. Personal users must instead migrate the project to Unity 2019.4.27 or newer and activate through Unity Hub, which changes the project's pinned-engine constraint and requires migration verification.

Official references:

- https://support.unity.com/hc/en-us/articles/211438683-How-do-I-activate-my-license
- https://support.unity.com/hc/en-us/articles/4401914348436-How-do-I-manually-activate-my-Unity-license
- https://docs.unity.com/en-us/hub/manage-license
- https://docs.unity3d.com/2019.3/Documentation/Manual/CommandLineArguments.html
