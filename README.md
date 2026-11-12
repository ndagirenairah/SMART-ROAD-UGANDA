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

## How to see the model annotating

The annotated videos are saved in the `outputs` folder. There are two ways to
watch them.

### Option 1: Watch it live while it runs

```powershell
python detect_track.py --input .\road.mp4 --direction north --imgsz 320 --show
```

A window opens showing boxes, trails, and labels as each frame is processed.
Press `q` to stop. It may be slow on a laptop because inference is running
live on the CPU.

Add separate output names if you do not want to overwrite saved results:

```powershell
python detect_track.py --input .\road.mp4 --direction north --imgsz 320 --show `
  --output-video .\outputs\live_north.mp4 `
  --output-csv .\outputs\live_north.csv
```

### Option 2: Open the saved videos

This is the recommended panel demonstration:

```powershell
start .\outputs\annotated_north.mp4
start .\outputs\annotated_south.mp4
```

- `annotated_north.mp4` is the real north-direction run.
- `annotated_south.mp4` is the reverse-direction test and must be presented
  as a **simulated diagnostic test**, not as evidence of real wrong-way
  driving.

If a video will not play, install VLC and open the files there. OpenCV's
`mp4v` format may not play correctly in the default Windows media player.

## What the prototype is for the panel

The prototype is not a web page. It consists of:

1. The script, `detect_track.py`
2. The annotated video it produces
3. The CSV containing the track evidence and labels
4. The GitHub repository

## What to look for in the video

- **Green** box and trail: moving in the permitted direction.
- **Red** box and trail: moving against the configured direction.
- **Yellow** box and trail: insufficient movement, so the track is not judged.
- Label text in the form `#ID class | classification`.

## Demo plan for the panel

1. Show `annotated_north.mp4` for 30 to 60 seconds.
2. Pause on a green vehicle, a red vehicle, and a yellow vehicle, and explain
   what each colour means.
3. Show `annotated_south.mp4` and explain: “This is the same video with the
   opposite reference direction; labels flip. This is a simulated test.”
4. Open `tracks_north.csv` to show the per-track evidence.
5. Show the GitHub repository.
6. Keep a recorded backup in case a live run is too slow.

The manual review count must come from watching the source and annotated
videos. Do not substitute the model's own labels for ground truth.
