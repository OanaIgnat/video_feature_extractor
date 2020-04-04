import glob


def create_input_csv():
    video_files = []
    for file in glob.glob("../S3D_HowTo100M/videos/*.mp4"):
        video_files.append(file)

    for file in glob.glob("../S3D_HowTo100M/videos/*.webm"):
        video_files.append(file)

    with open('input.csv', 'w+') as file:
        file.write("video_path,feature_path")
        for video_path in video_files:
            line = video_path + "," + "output/"+ video_path.split("/")[-1].split(".")[0]+".npy"
            file.write(line)
            file.write('\n')

create_input_csv()