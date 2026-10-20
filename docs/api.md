# Public API and compatibility

From 1.0.0, mj_viser follows [Semantic Versioning](https://semver.org/) for the public API
listed here. Everything else (other modules, other names, anything beginning with `_`) is
internal and may change in any release.

## The public API

A name is public when it is listed below. For a class, that includes its documented
constructor arguments, methods, properties, and attributes not beginning with `_`; for a
function, its documented arguments and return value; for a protocol, what an
implementation must provide.

### Core

Core needs no ssrobot. Importing `mj_viser`, or any core module, never imports it.

- `mj_viser.MujocoViewer`
- `mj_viser.PanelBase`
- `mj_viser.SensorPanel`
- `mj_viser.SensorChannel`
- `mj_viser.LiveView`
- `mj_viser.RenderSource`
- `mj_viser.RenderFrame`
- `mj_viser.SourceStale`
- `mj_viser.ChannelBoard`
- `mj_viser.ChannelState`
- `mj_viser.ContactOverlay`
- `mj_viser.contact_set`
- `mj_viser.MotionPreview`
- `mj_viser.JointMotion`
- `mj_viser.Highlight`
- `mj_viser.StateRecorder`

And in their modules, for the values those return or take:

- `mj_viser.contacts.Contact`
- `mj_viser.contacts.ContactSet`
- `mj_viser.contacts.LIMIT`
- `mj_viser.preview.record_motion`
- `mj_viser.recording.TEXT`
- `mj_viser.recording.IMAGES`
- `mj_viser.recording.CONTACTS`

### The ssrobot integration

`mj_viser.integrations.ssrobot` needs ssrobot, installed with its MuJoCo extra; without it,
importing the integration raises `ModuleNotFoundError` saying so.

- `mj_viser.integrations.ssrobot.render_source`
- `mj_viser.integrations.ssrobot.TraceChannels`
- `mj_viser.integrations.ssrobot.PlanPreview`
- `mj_viser.integrations.ssrobot.record_preview`
- `mj_viser.integrations.ssrobot.replay.record_trace`
- `mj_viser.integrations.ssrobot.replay.convert_bundle`
- `mj_viser.integrations.ssrobot.replay.Replay`
- the command `python -m mj_viser.integrations.ssrobot.replay BUNDLE OUT_DIR`

The integration's names and signatures follow the same versioning. The ssrobot it works
with does not: ssrobot is not yet 1.0, and each mj_viser release supports the ssrobot commit
its CI pins (see the release notes). A change in ssrobot can require an mj_viser release;
it is a breaking change of mj_viser only if it changes the names above.

### Behavior that is part of the API

- The viewer is read-only: it never steps, resets, or mutates the caller's `MjData`.
- Scene paths are built from MuJoCo ids and names (see `MujocoViewer.scene_report()`), and
  are stable for a given model across minor and patch releases.
- `scene_report()`'s keys and the meaning of their values.
- A `.viser` recording is Viser's format; mj_viser promises what it records (states at the
  times given, the text, images, and contacts), not the bytes.

Reports written for people (a replay's `conversion.json`, captions, panel text) are
informational: their content may grow or change wording in a minor release.

## Versioning

- **Patch** (`1.0.x`): fixes that do not change the public API.
- **Minor** (`1.x.0`): new names, arguments, or behavior that existing code does not
  notice; deprecations; support for new Python, MuJoCo, or Viser versions.
- **Major** (`x.0.0`): removing or renaming a public name, removing or reordering an
  argument, or changing documented behavior.

Raising the minimum Python or a dependency's minimum is done in a minor release and noted
in the changelog; mj_viser supports the Python versions every dependency has wheels for.

## Deprecation

A public name or argument is removed only in a major release, after at least one minor
release in which using it emits a `DeprecationWarning` naming its replacement. The
changelog lists every deprecation and removal.

## Versions and tags

A release's version is the `version` in `pyproject.toml`, and its tag is that version
prefixed with `v`: `v1.2.0` for a release, `v1.2.0rc1` for a release candidate. The release
workflow refuses to publish when the tag is not annotated, does not match the package's
version exactly, or names a version the index already has. PyPI never accepts a version
twice, so a published version cannot be replaced; a fix is a new version.
