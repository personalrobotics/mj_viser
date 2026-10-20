# README clips

The clips at the top of the project README are real mj_viser sessions, recorded in a
headless browser as they run. `capture.py` makes all of them; this file says from what.

| Clip | Shows | Inputs |
| --- | --- | --- |
| `geodude_contacts` | An ssrobot run followed live: Geodude plans to the cylinder's side, reaches, and closes its hand, with a `ContactOverlay` | ssrobot reference scene `geodude-cylinder.xml`; contact channels for the cylinder against each finger |
| `ada_live` | ADA's channels live: `TraceChannels` panels with the `arm_q` plot and the `wrist_rgb` image while the arm sweeps | ssrobot reference scene `ada-cylinder.xml` (with the seated user) |
| `geodude_preview` | `PlanPreview`: a browser drags the sample slider across the planned reach's trajectory | The same plan as `geodude_contacts` |
| `geodude_replay` | The same grasp saved with `ssrobot.write_bundle`, converted by `python -m mj_viser.integrations.ssrobot.replay`, and played in Viser's player | The same plan and scene |
| `g1` | Raw MuJoCo: the caller's loop poses Unitree G1 from its `stand` keyframe into a wave | MuJoCo Menagerie `unitree_g1/scene_with_hands.xml` |
| `shadow_hand` | Raw MuJoCo: Shadow Hand interpolated through its own keyframes | MuJoCo Menagerie `shadow_hand/scene_right.xml` and `keyframes.xml` |

## Reproducing them

Pinned inputs, all public:

- MuJoCo Menagerie at `feadf76d42f8a2162426f7d226a3b539556b3bf5`, the commit CI renders, with
  `unitree_g1` and `shadow_hand` checked out;
- ssrobot at `116f3c8214a6b3cc297a640a0ddf622c6fe497db` (the commit CI's integration lane pins),
  with its MuJoCo and planning extras, and its `references/` directory;
- geodude_assets and ada_assets at the commits that ssrobot's `references/robots.toml` pins;
- mj_viser from this checkout;
- Playwright (Python) with Google Chrome, and ffmpeg with libwebp.

```bash
git init -q menagerie && cd menagerie
git remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git sparse-checkout set unitree_g1 shadow_hand
git fetch -q --depth 1 --filter=blob:none origin feadf76d42f8a2162426f7d226a3b539556b3bf5
git checkout -q FETCH_HEAD && cd ..
git clone -q https://github.com/personalrobotics/ssrobot.git && git -C ssrobot checkout -q 116f3c8

uv venv --python 3.12 media-env
uv pip install --python media-env/bin/python -e . playwright \
  "ssrobot[mujoco,planning] @ git+https://github.com/personalrobotics/ssrobot.git@116f3c8214a6b3cc297a640a0ddf622c6fe497db" \
  "geodude-assets @ git+https://github.com/personalrobotics/geodude_assets.git@4f57fc45508c5dcad0c2426f48bb046d499340d0" \
  "ada-assets @ git+https://github.com/personalrobotics/ada_assets.git@b9880d45457dde90ae11f3704b6d8b1b6e99d00d"
media-env/bin/python -m playwright install ffmpeg

PATH="$PWD/media-env/bin:$PATH" media-env/bin/python -I scripts/media/capture.py \
  menagerie ssrobot/references media-out   # or name clips: ... media-out g1 ada_live
cp media-out/*.webp media-out/*.jpg assets/readme/
```

Each clip becomes `<clip>.webp`, an animated WebP at 15 fps (640 px wide; `ada_live` 800 px,
so its panels stay legible), and `<clip>.jpg`, a still at 80% of the clip that the README
shows when a reader prefers reduced motion. Plans come from a sampling planner, so a new
capture shows the same scene and motion kind but not byte-identical frames.

## Size budget

At most 500 KiB per animated clip and 2 MiB for all of `assets/readme/`; the current set is
about 1.2 MiB. Prefer a shorter or tighter clip over a lower quality setting.

The README refers to the clips by relative paths. PyPI shows the same README but resolves
no relative paths, so the build rewrites them to `raw.githubusercontent.com` URLs on `main`;
`scripts/check_readme.py` checks that every one of those files exists.
