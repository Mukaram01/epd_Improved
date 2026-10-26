# EPD in the studio design language

EPD shares its design language with Workcell Studio. The full specification is
`easy_manipulation_deployment/docs/manuals/WORKCELL_STUDIO_DESIGN_LANGUAGE.md`. EPD keeps its dark
perception-console identity; `gui/windows/studio_design.py` is the single place that aligns it with
Workcell Studio.

`main.build_main_window()` applies this layer last, after every productization pass.

## What the layer does

- **Shared tokens.** Same names and dark values as the Builder's `studioTokens()`. The primary
  action (Run Perception) uses the shared accent `#246FE5`; the earlier violet was the only
  off-family colour.
- **Application base.** Thin dark scrollbars (replacing the native white ones), tooltips, keyboard
  focus outline and splitter hover.
- **Quiet readiness.** Legacy code painted whole Model, Labels and Mode buttons saturated green or
  red. That state is now a `ready` property, shown as a danger border when not ready. The compact
  "✓ Ready" chips remain the primary signal.
- **Readable type.** Every stylesheet `font-size` is floored at 10 px; 9 px captions become 10 px.
  **View → Text size** (100 / 112 / 125 %) rescales live from the original values, without
  compounding.
- **Unplaced widgets hidden.** Unnamed labels and buttons left out of every layout by the layered
  refreshes (for example the old "Confidence Threshold" labels) are hidden. They painted behind
  other widgets and still took keyboard focus.
- **Layout memory.** Named splitters (`monitorMainSplitter`, `monitorLeftSplitter`) are restored.
  **View → Reset Layout** (Ctrl+Shift+0) restores the default panel and window sizes.
- **Messages.** "No problems reported." is neutral; amber is used only when there are messages.

## Checking it

```bash
cd gui
python3 scripts/ui_tour_capture.py --out /tmp/epd_ui --size 1366x768 --tag 1366x768
```

The tour builds the real GUI and captures the launcher, Deploy and each of its tabs, Train and Help.
It writes an audit of clipped text and horizontal overflow. Set `QT_SCALE_FACTOR` for scaled
checks, and use an isolated `XDG_CONFIG_HOME`. Starting the GUI rewrites
`config/session_config.json` and `usecase_config.json` (formatting and schema version only), as it
always has.
