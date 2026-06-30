# Science Birds Xvnc Frame Capture

Date: 2026-06-22

Useful runtime finding: on this migrated server, the Java/Unity Science Birds engine can start and reach the protocol `PLAYING` state, but the TCP screenshot protocol can still return uniform gray frames. `Xvfb` from conda does not expose GLX here. `Xvnc` does expose GLX and lets Unity create an OpenGL context via Mesa llvmpipe.

Working display setup:

```sh
source ~/cd_novphy
Xvnc :132 -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >/tmp/novphy_xvnc.log 2>&1 &
export DISPLAY=:132
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"
```

Packages installed into the `novphy` conda env during validation:

- `xorg-xserver-xvfb`
- `mesalib`
- `xorg-x11-server-xvfb-conda-x86_64`
- `xdotool`

Validated capture path:

1. Start `sciencebirdsgames/Linux/game_playing_interface.jar --dev` on the same `DISPLAY`.
2. Connect/configure through `src.webui.bridge.ScienceBirdsBridge`.
3. Use `scripts.manual_agent.prepare_for_play()` to get the backend ready.
4. The visible Unity window may still show the title screen, so use `xdotool` on the same display:
   - click Play around `(512, 390)` on a 1024x768 Xvnc screen,
   - click level input around `(495, 343)`,
   - type `1`,
   - click confirm around `(492, 465)`.
5. Capture desktop frames with `PIL.ImageGrab.grab()`.

Verified artifact from this server:

- Manifest: `data/migration_engine_check_20260622_0348_xvnc_level1_confirm/manifest.json`
- Frames: `data/migration_engine_check_20260622_0348_xvnc_level1_confirm/frames/frame_0000.png` through `frame_0014.png`
- Visual check: `frame_0014.png` shows in-level gameplay with slingshot, birds, pig, score UI, and level scene.
- Frame stats: all 15 frames have `uniform: false`; final frame has `unique_colors: 3847`.

Do not put sudo passwords into commands or logs. If system packages are needed later, ask the user to run the apt command themselves or use an interactive secure channel outside logged tool arguments.
