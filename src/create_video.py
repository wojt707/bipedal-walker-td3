from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    TextClip,
    CompositeVideoClip,
)
import os
import json


def add_episode_number(video_path: str, episode_number: int):
    video_clip = VideoFileClip(video_path)
    text_clip = (
        TextClip(
            f"Episode {episode_number}", fontsize=24, color="black", font="Robotic"
        )
        .set_duration(video_clip.duration)
        .set_position(("left", "top"))
    )
    return CompositeVideoClip([video_clip, text_clip])


def create_final_video(video_folder: str, should_delete_clips: bool = False):
    # get list of json files
    json_files: list[str] = [
        f for f in os.listdir(video_folder) if f.endswith(".meta.json")
    ]

    # tuples [video_file_name, episode_number_of_this_video]
    videos: list[tuple[str, int]] = []

    for json_file in json_files:
        video_file = json_file.replace(".meta.json", ".mp4")
        json_path = os.path.join(video_folder, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)
            videos.append((video_file, data["episode_id"]))

    videos.sort(key=lambda x: x[1])

    clips = []
    for video_file, episode_number in videos:
        video_path = os.path.join(video_folder, video_file)
        clips.append(add_episode_number(video_path, episode_number))

    final_path = os.path.join(video_folder, "final.mp4")
    final_clip = concatenate_videoclips(clips)

    try:
        final_clip.write_videofile(final_path, codec="libx264", threads=6, logger=None)
        print(f"Saved .mp4 without Exception at {final_path}")
    except IndexError:
        # Short by one frame, so get rid on the last frame:
        final_clip = final_clip.subclip(
            t_end=(final_clip.duration - 1.0 / final_clip.fps)
        )
        final_clip.write_videofile(final_path, codec="libx264", threads=6, logger=None)
        print(f"Saved .mp4 after Exception at {final_path}")
    except Exception as e:
        print("Exception {} was raised!!".format(e))

    if should_delete_clips:
        for file in os.listdir(video_folder):
            file_path = os.path.join(video_folder, file)
            if file != "final.mp4":
                os.remove(file_path)


# create_final_video(os.path.join("saved", "BipedalWalker-v3_0", "videos"))
