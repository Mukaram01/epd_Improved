# EPD Profiles & Replay

EPD-5 makes a perception setup reproducible instead of treating the three deploy configuration files as unrelated state.

Open **Deploy Perception** and choose **Profiles & Replay**.

## Perception profiles

A profile captures the deploy configuration that affects the operator-visible perception session:

- ONNX model path;
- label-list path;
- detection overlay mode;
- CPU/GPU selection;
- ONNX Runtime thread setting currently stored by EPD;
- image transport;
- object-mask publishing;
- confidence threshold;
- maximum detections;
- perception/use-case mode and mode-specific options;
- RGB camera topic.

Profiles use `profile_schema_version: 1` and contain snapshots of:

- `session_config.json`;
- `usecase_config.json`;
- `input_image_topic.json`.

They are stored by default under:

```text
~/.config/WorkcellStudio/EasyPerceptionDeployment/profiles/
```

This keeps personal/runtime profiles out of the Git checkout and prevents normal `git pull` operations from treating them as source changes.

## Asset fingerprints

When the selected ONNX model and label list exist, EPD records a SHA256 digest and file size in the profile.

When the profile is applied later, EPD checks those fingerprints. If the original configured path no longer exists, EPD may relocate an asset to the local package `data/model/` or `data/label_list/` folder by basename, but only when the captured hash matches.

EPD refuses to silently substitute a same-named but different model or label file.

## Save current

Configure Deploy normally, then choose **Save current**. Give the profile a useful operational name such as:

```text
D435i Table Pick - MaskRCNN v3
```

Use the description for information not encoded in EPD, for example:

- workcell/camera mounting position;
- part family;
- lighting condition;
- acceptance date;
- related Workcell Studio scene name.

Do not put passwords, API keys or other secrets in profile descriptions.

## Apply selected

Select a profile and choose **Apply selected**.

EPD-5:

1. validates the profile envelope;
2. resolves and verifies model/label assets;
3. atomically writes the three deploy configuration files;
4. updates the visible Deploy controls;
5. re-runs the normal deploy readiness/model inspection path.

Profile changes are blocked while a deployment is STARTING, RUNNING or STOPPING. Stop perception before changing profile.

## Known-good profile

Select a validated profile and choose **Set known-good**.

**Restore known-good** then becomes the fast recovery action for returning the workstation to the chosen configuration.

The marker is local to the workstation. Export the actual profile if it should be shared with another PC.

## Import / export

Profiles are normal JSON files and can be exported for review, source-controlled in a separate deployment/config repository, attached to acceptance evidence, or imported on another EPD workstation.

Importing does not bypass asset validation. Missing/mismatched model or label files still block application.

## Deterministic fixture replay

The Replay tab exposes the existing EPD deterministic replay pipeline:

```text
ros2 launch easy_perception_deployment replay.launch.py
```

Select a fixture JSON and choose either:

- **fast** — deliver observations as quickly as the production pipeline accepts them;
- **realtime** — preserve fixture timestamp spacing.

The replay launch starts its own EPD production node, so stop the normal Deploy session first.

At EOF, the existing replay acceptance summary is loaded into the GUI. The result remains the backend's `PASS` or `FAIL`; the GUI does not reinterpret it.

The bundled P8 fixture is selected by default when available.

## Rosbag2 replay

Rosbag replay is deliberately conservative.

1. Apply the profile that corresponds to the recording.
2. Start Deploy normally.
3. Choose the rosbag2 directory.
4. Select **Inspect bag**.
5. Confirm the active profile RGB topic appears in the recorded topic list.
6. Select **Play bag**.

EPD-5 invokes:

```text
ros2 bag play <bag-directory>
```

It does **not** invent or silently apply topic remappings. This is intentional: a replay should reproduce the recorded interface, not hide a mismatch between the profile and the bag.

For Localization and Tracking, the bag must also provide the depth/CameraInfo streams expected by the running EPD configuration. Use Camera Assistant and the live diagnostics to verify those streams.

## What a profile does not own

EPD profiles do not contain:

- Workcell Studio scene geometry;
- robot poses;
- MoveIt planning settings;
- grasp/task definitions;
- real-hardware authorization.

That separation is deliberate. EPD owns perception configuration; Workcell Studio / EMD own scene/task/planning concerns.

## Recommended acceptance workflow

For a workcell perception setup:

1. configure camera/model/mode;
2. run Camera Assistant;
3. run Smart Model Manager inspection;
4. verify Live Perception View;
5. perform the relevant deterministic/recorded replay test;
6. run the live acceptance test;
7. save the profile;
8. mark it known-good only after the acceptance evidence is satisfactory;
9. export the profile with the workcell acceptance artefacts.

This gives the next Workcell Studio integration phase a stable profile reference instead of a loose collection of GUI settings.
