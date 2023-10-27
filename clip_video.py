import os
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
from moviepy.editor import VideoFileClip
from pathlib import Path

def extract_frames(video_path, num_frames):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame_times = [duration * i / (num_frames + 1) for i in range(1, num_frames + 1)]
    
    frames = []
    for t in frame_times:
        frame = clip.get_frame(t)
        image = Image.fromarray(frame)
        frames.append(image)
    
    clip.close()
    
    return frames

def save_frames(video_file):
    num_frames = 12  # define your number of frames here
    frames = extract_frames(video_file, num_frames)
    for i, frame in enumerate(frames):
        save_dir = video_file.replace(ori_path, target_path)
        save_dir = os.path.dirname(save_dir)  
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        frame.save(f"{save_dir}/{Path(video_file).stem}_{i}.jpg")

# get funqa,msvtt,msvttqa video clips
def main(dir):
    video_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))
    
    with Pool(os.cpu_count()) as p:
        list(tqdm(p.imap(save_frames, video_files), total=len(video_files)))

if __name__ == "__main__":
    dir = '.data/funqa'
    # dir = '.data/msrvtt'
    ori_path = '.data/funqa'
    # ori_path = '.data/msrvtt'
    target_path = ".data/funqa_cliped"
    # target_path = ".data/video_cliped"
    main(dir)
