from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
    api = YouTubeTranscriptApi()
    return api.fetch(video_id, languages=['en'])


def get_video_id(url):
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")


video_url = "https://www.youtube.com/watch?v=qfsjoPzbAxQ"

video_id = get_video_id(video_url)

transcript = get_transcript(video_id)

document = "\n".join([entry.text for entry in transcript])  

def documents():
    return document
