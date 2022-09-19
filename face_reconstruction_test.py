import os
import cv2
import time
import torch
import numpy as np
from argparse import ArgumentParser
from ibug.face_alignment import FANPredictor
from ibug.face_alignment.utils import plot_landmarks
from ibug.face_detection import RetinaFacePredictor
from ibug.face_reconstruction import DecaCoarsePredictor


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--input", "-i", default=0,
        help="Input video path or webcam index (default=0)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output filename and path",
    )
    parser.add_argument(
        "--fourcc", "-f", type=str, default="mp4v", 
        help="FourCC of the output video (default=mp4v)",
    )
    parser.add_argument(
        "--benchmark", "-b", action="store_true", default=False,
        help="Enable benchmark mode for CUDNN",
    )
    parser.add_argument(
        "--no-display", "-n", action="store_true", default=False,
        help="No display if processing a video file",
    )
    # arguments for face reconstruction
    parser.add_argument(
        '--reconstruction-weights', '-rw', default='ar_res50_coarse',
        help='Weights to be loaded for face reconstruction, i.e., ar_res50_coarse, ar_mbv2_coarse, flame_res50_coarse or flame_mbv2_coarse'
    )
    parser.add_argument(
        "--reconstruction-device", "-rd", default="cuda:0",
        help="Device to be used for face reconstruction (default=cuda:0)",
    )
    parser.add_argument(
        "--show-reconstruction-bbox", "-srb", action="store_true", default=False,
        help="Do not visualise bbox for face reconstruction",
    )
    parser.add_argument(
        "--show-reconstruction-landmarks2d", "-sr2d", action="store_true", default=False,
        help="Do not visualise 2D-style landmarks from face reconstruction",
    )
    parser.add_argument(
        "--hide-reconstruction-pose", "-hrp", action="store_true", default=False,
        help="Do not visualise estimated pose from face reconstruction",
    )
    parser.add_argument(
        "--hide-reconstruction-landmarks3d", "-hr3d", action="store_true", default=False,
        help="Do not visualise 3D landmarks from face reconstruction",
    )
    # arguments for face detection
    parser.add_argument(
        "--detection-threshold", "-dt", type=float, default=0.95,
        help="Confidence threshold for face detection (default=0.95)",
    )
    parser.add_argument(
        "--detection-device", "-dd", default="cuda:0",
        help="Device to be used for face detection (default=cuda:0)",
    )
    # arguments for face alignment
    parser.add_argument(
        "--alignment-device", "-ad", default="cuda:0",
        help="Device to be used for face alignment (default=cuda:0)",
    )
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    vid = None
    out_vid = None
    has_window = False
    try:
        # Create the RetinaFace detector 
        fd_model = RetinaFacePredictor.get_model()
        face_detector = RetinaFacePredictor(
            threshold=args.detection_threshold, device=args.detection_device, model=fd_model,
        )

        # Create the 2D FAN landmark detector
        fa_model = FANPredictor.get_model("2dfan2_alt")
        landmark_detector = FANPredictor(device=args.alignment_device, model=fa_model)

        # Create the DECA coarse reconstructor 
        frec_model = DecaCoarsePredictor.create_model_config(args.reconstruction_weights)
        face_reconstructor = DecaCoarsePredictor(
            device=args.reconstruction_device, model_config=frec_model,
        )

        # Open the input video
        using_webcam = not os.path.exists(args.input)
        vid = cv2.VideoCapture(int(args.input) if using_webcam else args.input)
        assert vid.isOpened()
        if using_webcam:
            print(f"Webcam #{int(args.input)} opened.")
        else:
            print(f"Input video {args.input} opened.")

        # Open the output video (if a path is given)
        if args.output is not None:
            out_vid = cv2.VideoWriter(args.output, fps=vid.get(cv2.CAP_PROP_FPS),
                                      frameSize=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      fourcc=cv2.VideoWriter_fourcc(*args.fourcc))
            assert out_vid.isOpened()

        # Process the frames
        frame_number = 0
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        print("Processing started, press \"Q\" to quit.")
        while True:
            # Get a new frame
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Detect faces
                start_time = time.time()
                faces = face_detector(frame, rgb=False)
                face_detection_time = time.time() - start_time

                # Face alignment
                start_time = time.time()
                landmarks, scores = landmark_detector(frame, faces, rgb=False)
                face_alignment_time = time.time() - start_time

                # Face reconstruction
                start_time = time.time()
                results = face_reconstructor(frame, landmarks, rgb=True)
                face_reconstruction_time = time.time() - start_time

                # Textural output
                total_time = 1000.0 * sum([face_detection_time, face_alignment_time, face_reconstruction_time])
                print(f"Frame #{frame_number} processed in {total_time:.02f} ms, {len(faces)} faces analysed.")
                print(f"3D face reconstruction took {face_reconstruction_time * 1000.0:.02f} ms.")

                # Rendering
                if results is not None:
                    n_faces = results["vertices"].shape[0]
                    for idx in range(n_faces):
                        if args.show_reconstruction_bbox:
                            bbox = results["bboxes"][idx].astype(int)
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
                        if args.show_reconstruction_landmarks2d:
                            landmarks2d = results["landmarks2d"][idx]
                            plot_landmarks(frame, landmarks2d, line_colour=(255, 0, 0))
                        if not args.hide_reconstruction_landmarks3d:
                            landmarks3d = results["landmarks3d"][idx, :, :2]
                            plot_landmarks(frame, landmarks3d)
                        if not args.hide_reconstruction_pose:
                            # TODO: to check if yaw, picth and roll's sign comply with the BC internal 
                            yaw, pitch, roll = results["face_poses"][idx] * 180. / np.pi
                            bbox = results["bboxes"][idx].astype(int)
                            frame_diagonal = np.linalg.norm(frame.shape[:2])
                            font_scale = max(0.3, frame_diagonal / 2000.)
                            thickness = int(max(1, np.round(2.0 * font_scale)))
                            cv2.putText(
                                frame,
                                f"Yaw:{int(yaw)}  Pitch:{int(pitch)}  Roll:{int(roll)}",
                                (bbox[0] - 100, bbox[3] + int(0.15*(bbox[3]-bbox[1]))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale,
                                color=(0, 0, 180),
                                thickness=thickness,
                            )

                # Write the frame to output video (if recording)
                if out_vid is not None:
                    out_vid.write(frame)

                # Display the frame
                if using_webcam or not args.no_display:
                    has_window = True
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1) % 2 ** 16
                    if key == ord("q") or key == ord("Q"):
                        print("\"Q\" pressed, we are done here.")
                        break
                frame_number += 1
    finally:
        if has_window:
            cv2.destroyAllWindows()
        if out_vid is not None:
            out_vid.release()
        if vid is not None:
            vid.release()
        print("All done.")


if __name__ == "__main__":
    main()
