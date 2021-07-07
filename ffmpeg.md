# Speeding up video (audio unchanged) by changing presentation timestamp (PTS) of each video frame
`ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" output.mp4`
