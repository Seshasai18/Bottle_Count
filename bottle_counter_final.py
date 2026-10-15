import os
import cv2
from ultralytics import YOLO
MODEL_PATH = r"D:\Resolute Office Works\Extracted_Frames\best (1).pt"
VIDEO_PATH = 
OUTPUT_PATH = r"D:\Resolute Office Works\Extracted_Frames"

CONF = 0.15
LINE_POSITION = 0.5
SHOW_PREVIEW = True  

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

    model = YOLO(MODEL_PATH)
    print("Model classes:", model.names)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {VIDEO_PATH}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_x = int(w * LINE_POSITION)

    count = 0
    counted_ids = set()
    prev_side = {}
    frame_idx = 0

    print("Processing video... press 'q' in the preview window to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        results = model.track(frame, persist=True, conf=CONF,
                               tracker="bytetrack.yaml", verbose=False)
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                cur_side = -1 if cx < line_x else 1
                last_side = prev_side.get(tid)
                if last_side is not None and last_side != cur_side and tid not in counted_ids:
                    count += 1
                    counted_ids.add(tid)
                prev_side[tid] = cur_side

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID{tid} {conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.line(frame, (line_x, 0), (line_x, h), (255, 0, 0), 3)
        cv2.putText(frame, f"Bottles counted: {count}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(frame)

        if SHOW_PREVIEW:
            cv2.imshow("Bottle Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n===== DONE =====")
    print("Frames processed:", frame_idx)
    print("TOTAL BOTTLES COUNTED:", count)
    print("Output video saved to:",r"D:\Resolute Office Works\Extracted_Frames")


if __name__ == "__main__":
    main()
