# Speeding up video (audio unchanged) by changing presentation timestamp (PTS) of each video frame
`ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" output.mp4`

# Speeding up video with audio
` ffmpeg -i input.mkv -filter_complex "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]" -map "[v]" -map "[a]" output.mkv`
