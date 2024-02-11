# def detect_tracks(video: Video, initial_bg: int, stop: int = None) -> List[Track]:
#     had_tracks = False
#     tracks: List[Track] = []
#     bg = prepare(video.read_frame_at(initial_bg).pixels)
#     for frame in video.iter_frames(start=initial_bg + 1, stop=stop):
#         # print(f"Processing {frame}")
#         thresh, binary = process_frame(frame, bg)
#         # print(f"Threshold: {thresh}")
#         if has_tracks(thresh):
#             had_tracks = True
#             # print("Tracks detected")
#             contours = find_tracks(binary)
#             update_tracks(tracks,
#                           join_close_contours(contours) if len(contours) > 1 else contours,
#                           frame, binary)
#         else:
#             # print("No tracks detected")
#             if had_tracks:
#                 print("Changing BG")
#                 bg = prepare(frame.pixels)
#             had_tracks = False
#     return tracks
