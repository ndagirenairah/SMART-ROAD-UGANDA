# SmartRoad UG research prototype

This project contains a real, local video-processing pipeline. The prototype
is intentionally demonstrated from the terminal and generated output files;
there is no browser dashboard that could be mistaken for an inference system.

## Run the computational prototype

1. Create and activate a Python environment.
2. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Process an actual road video:

   ```powershell
   python detect_track.py --input .\road.mp4 --direction north `
     --output-video .\outputs\annotated_north.mp4 `
     --output-csv .\outputs\tracks_north.csv
   ```

The first run downloads the selected Ultralytics weights if they are not
already present. The command writes:

- `outputs\annotated_north.mp4`: detections, ByteTrack IDs, trajectories, and the
  direction classification drawn on every processed frame.
- `outputs\tracks_north.csv`: one evidence row per observed track, including its
  movement angle, permitted direction, classification, and mean confidence.

For the reverse-direction test, use separate output names:

```powershell
python detect_track.py --input .\road.mp4 --direction south `
  --output-video .\outputs\annotated_south.mp4 `
  --output-csv .\outputs\tracks_south.csv
```

Use `--imgsz 320` on a memory-constrained CPU. It changes inference
resolution, not the dimensions of the saved annotated video.

The direction is expressed in image coordinates: `north` means upward in the
video, `east` means right, `south` means downward, and `west` means left.
Classification is based on the displacement between the first and latest
tracked centers. Tracks with too little displacement remain explicitly marked
as `insufficient track` or `insufficient movement`.

## Manual review

The rule-based labels are not ground truth. Select ten red (`wrong-way`) and
ten green (`normal direction`) tracks from an annotated output, compare each
with the source movement, and record the result in
`manual_review_template.csv`. Do not report an accuracy figure until those
cells have been completed.

## Current run summary

These are model output counts, not accuracy measurements. On the supplied
`road.mp4` at `--imgsz 320`, the north run produced 57 tracks: 32 normal
direction, 14 wrong-way, and 11 insufficient movement. The reverse `south`
run produced the same 57 tracks: 3 normal direction, 43 wrong-way, and 11
insufficient movement. The direction flip is a diagnostic comparison; it is
not proof that the clip is a one-way road.

## Reused versus ours

Reused: Ultralytics YOLO weights and inference API, ByteTrack integration,
OpenCV video I/O, and Python's CSV utilities.

Ours: the three-class selection, image-coordinate direction reference,
track-displacement rule, insufficient-movement state, per-track evidence CSV,
annotation colours, reverse-direction protocol, and manual review procedure.
