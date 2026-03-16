# Rollouts
Rollouts will be stored in this directory.

## Instructions
Nothing to do here :)

## After simulation
1. By default docker will create files which belong to `root`. A useful command to bring them back to current user: `sudo find . -type d -user root -exec sudo chown -R $USER: {} +` (assumes this is your current directory).
2. You can execute `./make-videos .` to recursively find all directories with `.png` images and convert them to videos with `ffmpeg`.
