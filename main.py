# main.py  — full, copy-paste file
import os
import re
import argparse
import json
import textwrap
from pathlib import Path
import time

# Agent framework + LLM
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# TTS + video + images
from gtts import gTTS
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from PIL import Image, ImageDraw, ImageFont

# YouTube uploader libs
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Load env
load_dotenv()

# LLM setup
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7))
)

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

# ---------- Helpers ----------
def save_output(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------- Agents ----------
researcher_agent = Agent(role="YouTube Researcher", goal="Research video ideas around a given keyword.", backstory="Expert in identifying trending topics on YouTube.", allow_delegation=False, llm=llm)
scriptwriter_agent = Agent(role="Script Writer", goal="Write a clear and engaging script for a YouTube video based on a given keyword.", backstory="Creative storyteller and video script writer.", allow_delegation=False, llm=llm)
seo_agent = Agent(role="SEO Specialist", goal="Optimize titles, description, tags, and hashtags strictly based on the given keyword.", backstory="SEO expert for YouTube channels.", allow_delegation=False, llm=llm)
thumbnail_agent = Agent(role="Thumbnail Designer", goal="Generate a thumbnail design prompt with overlay text for the given keyword.", backstory="Expert in designing eye-catching YouTube thumbnails.", allow_delegation=False, llm=llm)
tts_agent = Agent(role="Narrator", goal="Convert script text into voice narration MP3", backstory="Expert in creating engaging voice-overs from scripts", allow_delegation=False, llm=llm)
video_agent = Agent(role="Video Creator", goal="Generate a simple slideshow video with narration", backstory="Expert in creating videos using slides and narration", allow_delegation=False, llm=llm)

# ---------- Tasks builders ----------
def create_tasks(keyword: str):
    research_task = Task(
        description=(
            f"Research 5 YouTube video ideas around the keyword: '{keyword}'. "
            "Pick the most promising idea. Ensure ideas are directly related to the keyword."
        ),
        agent=researcher_agent,
        inputs={"keyword": keyword},
        expected_output=str(OUT_DIR / "research.md")
    )

    script_task = Task(
        description=(
            f"Write a ~220 word engaging YouTube script for the topic strictly based on the keyword: '{keyword}'. "
            "The script should be conversational and informative, avoid drifting off-topic."
        ),
        agent=scriptwriter_agent,
        inputs={"keyword": keyword},
        expected_output=str(OUT_DIR / "script.md")
    )

    seo_task = Task(
        description=(
            f"Based ONLY on the keyword '{keyword}' and the outputs of research + script, "
            "generate YouTube SEO metadata. Return JSON with 3 click-worthy titles (must contain the keyword), "
            "a 200-300 word description (keyword-rich), 10-12 tags (keyword-based), 3 hashtags, and a suggested schedule."
        ),
        agent=seo_agent,
        inputs={"keyword": keyword},
        expected_output=str(OUT_DIR / "metadata.json")
    )

    thumbnail_task = Task(
        description=(
            f"Generate 3 thumbnail prompt ideas strictly for the keyword '{keyword}'. "
            "Each should include a visual concept and suggested overlay text containing the keyword."
        ),
        agent=thumbnail_agent,
        inputs={"keyword": keyword},
        expected_output=str(OUT_DIR / "thumbnail_prompt.txt")
    )

    return [research_task, script_task, seo_task, thumbnail_task]

# ---------- TTS & Video functions ----------
def generate_tts(script_text: str, path: str = str(OUT_DIR / "narration.mp3")) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    tts = gTTS(text=script_text, lang="en")
    tts.save(path)
    return path

def generate_video(script_text: str, audio_path: str = str(OUT_DIR / "narration.mp3"), video_path: str = str(OUT_DIR / "final_video.mp4")) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Basic chunking for slides — you can replace with smarter chunking later
    # We'll use short paragraphs (split by sentences)
    #sentences = [s.strip() for s in textwrap.split(script_text) if s.strip()]
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', script_text) if s.strip()]
    if not sentences:
        sentences = textwrap.wrap(script_text, width=80)

    # safe: fallback to 5-chunk split
    if len(sentences) < 4:
        words = script_text.split()
        chunk_size = max(20, len(words) // 5)
        sentences = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    audio = AudioFileClip(audio_path)
    duration = audio.duration if hasattr(audio, "duration") else max(5, len(script_text.split()) / 3)

    # If too many slides, limit; otherwise we set per-slide duration proportional
    n_slides = min(max(3, len(sentences)), 8)
    per_slide = max(2.5, duration / n_slides)

    slides = []
    for i, text_block in enumerate(sentences[:n_slides]):
        txt_img = OUT_DIR / f"slide_{i}.png"
        create_slide_image(text_block, txt_img)
        # Depending on MoviePy version use with_duration or set_duration. We'll use with_duration which works with moviepy 2+
        clip = ImageClip(str(txt_img)).with_duration(per_slide)
        slides.append(clip)

    video = concatenate_videoclips(slides, method="compose")
    # Combine audio; allow audio shorter/longer — we'll trim/pad to video length
    video = video.with_audio(audio)
    video.write_videofile(video_path, fps=24, codec="libx264", audio_codec="aac")
    audio.close()
    return video_path

# slide image helper
def create_slide_image(text: str, out_path: Path, size=(1280, 720), bg=(30, 30, 30), fg=(255, 255, 255)):
    W, H = size
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    # Find font
    font = None
    for p in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            if Path(p).exists():
                font = ImageFont.truetype(p, 38)
                break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    margin = 80
    max_w = W - 2 * margin
    # naive wrap
    lines = textwrap.wrap(text, width=30)
    # center text block
    line_h = font.getsize("A")[1] + 10 if hasattr(font, "getsize") else 24
    block_h = line_h * len(lines)
    y = (H - block_h) // 2
    for line in lines:
        w = draw.textlength(line, font=font)
        x = (W - w) // 2
        draw.text((x, y), line, font=font, fill=fg)
        y += line_h

    img.save(out_path)
    return out_path

def safe_load_json(path: Path, retries: int = 4, delay: float = 0.8):
    """
    Try to load JSON from path. Retry a few times if file is empty or being written.
    Returns a dict (possibly empty) on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")
            # ensure file has some bytes (avoid 0-byte read)
            if path.stat().st_size == 0:
                raise ValueError("file is 0 bytes")
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError("file is empty")
            return json.loads(text)
        except Exception as e:
            print(f"⚠️ safe_load_json attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
    print("⚠️ safe_load_json failed after retries — returning empty metadata.")
    return {}


# ---------- Thumbnail generator (actual image) ----------
def generate_thumbnail(thumbnail_text: str, out_path: str = str(OUT_DIR / "thumbnail.jpg")) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), color=(40, 44, 52))
    draw = ImageDraw.Draw(img)
    # font selection
    font = None
    for p in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]:
        try:
            if Path(p).exists():
                font = ImageFont.truetype(p, 72)
                break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    # overlay text (wrap to lines)
    lines = textwrap.wrap(thumbnail_text, width=12)
    # draw a semi-transparent rectangle for contrast
    rect_h = 100 + 90 * len(lines)
    rect_y = (H - rect_h) // 2
    draw.rectangle([(60, rect_y), (W - 60, rect_y + rect_h)], fill=(10, 10, 10, 200))
    y = rect_y + 20
    for line in lines:
        w = draw.textlength(line, font=font)
        x = (W - w) // 2
        draw.text((x, y), line, font=font, fill=(255, 215, 0))
        y += 80
    img.save(out_path, quality=85)
    return out_path

# ---------- YouTube uploader helpers ----------
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def get_authenticated_service(client_secrets_file="client_secret.json", token_file="token.json"):
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secrets_file):
                raise FileNotFoundError(f"client_secret.json not found at {client_secrets_file}. Create OAuth credentials in Google Cloud Console.")
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        # save creds
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)

def upload_to_youtube(youtube_service, video_file: str, title: str, description: str, tags: list, thumbnail_file: str, privacyStatus: str = "private"):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "28"  # Science & Tech (reasonable for AI topics)
        },
        "status": {
            "privacyStatus": privacyStatus
        }
    }
    media = MediaFileUpload(video_file, chunksize=-1, resumable=True)
    request = youtube_service.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    # Do a resumable upload
    res = request.execute()
    video_id = res.get("id")
    print(f"Uploaded video ID: {video_id}")
    if thumbnail_file and os.path.exists(thumbnail_file):
        #youtube_service.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_file)).execute()
        print("Thumbnail uploaded.")
    return video_id

# ---------- Pipeline runner ----------
def run_pipeline(keyword: str, do_upload: bool = False):
    tasks = create_tasks(keyword)
    crew = Crew(
        agents=[researcher_agent, scriptwriter_agent, seo_agent, thumbnail_agent, tts_agent, video_agent],
        tasks=tasks,
        process=Process.sequential
    )
    print(f"▶ Running agents for keyword: {keyword}")
    crew.kickoff(inputs={"keyword": keyword})

    # Save outputs produced by tasks
    for t in tasks:
        out_path = t.expected_output
        # some agent frameworks put result in t.output or on disk; try to read from t.output if present
        if getattr(t, "output", None):
            content = str(t.output)
            save_output(out_path, content)
            print(f"Saved {out_path} (from task.output)")
        else:
            # nothing came from the agent? try to leave placeholder
            if not os.path.exists(out_path):
                save_output(out_path, f"(no direct task output; check agent logs for task {t.description})")
                print(f"Created placeholder {out_path}")

    # Read script content (required for TTS/video)
    script_path = OUT_DIR / "script.md"
    if not script_path.exists():
        raise FileNotFoundError("Script not found at out/script.md — check the script agent output.")
    script_text = read_text(str(script_path))

    # Generate TTS and video
    print("🎙 Generating narration...")
    narration_path = generate_tts(script_text)

    print("🎞 Generating video...")
    video_path = generate_video(script_text, narration_path)

    # Generate thumbnail image using the thumbnail prompt file (if present), else fallback to title
    thumb_prompt_path = OUT_DIR / "thumbnail_prompt.txt"
    if thumb_prompt_path.exists():
        thumb_text = read_text(str(thumb_prompt_path)).strip().splitlines()[0][:120]
    else:
        # fallback to reading metadata to get title
        md_path = OUT_DIR / "metadata.json"
        if md_path.exists():
            md = json.loads(read_text(str(md_path)))
            thumb_text = md.get("titles", ["Thumbnail"])[0]
        else:
            thumb_text = keyword
    print("🖼 Generating thumbnail image...")
    thumbnail_file = generate_thumbnail(thumb_text)

    print(f"✅ Video: {video_path}")
    print(f"✅ Thumbnail: {thumbnail_file}")

        # Optionally upload
    if do_upload:
        md_path = OUT_DIR / "metadata.json"
        # safe load with retries
        metadata = safe_load_json(md_path)

        # pick title: prefer first item in metadata["titles"] if present, else fallback
        title = None
        if isinstance(metadata, dict):
            titles = metadata.get("titles") or metadata.get("title")  # accept both shapes
            if isinstance(titles, list) and titles:
                title = titles[0]
            elif isinstance(titles, str) and titles.strip():
                title = titles
            else:
                # fallback: use the keyword as title
                title = keyword
        else:
            title = keyword

        description = metadata.get("description", "") if isinstance(metadata, dict) else ""
        tags = metadata.get("tags", []) if isinstance(metadata, dict) else []

        print(f"🔎 Upload metadata prepared — title: {title!r}, tags: {len(tags)} items")

        # Check OAuth credentials file
        client_secrets = Path("client_secret.json")
        if not client_secrets.exists():
            raise FileNotFoundError("client_secret.json not found. Put your OAuth credentials in the project root.")

        print("🔐 Authenticating to YouTube (OAuth)...")
        yt = get_authenticated_service(client_secrets_file="client_secret.json", token_file="token.json")
        print("📤 Uploading to YouTube (privacy=private while testing)...")
        video_id = upload_to_youtube(yt, video_path, title, description, tags, thumbnail_file, privacyStatus="private")
        print(f"✅ Uploaded: https://youtu.be/{video_id}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True, help="Keyword for YouTube content generation")
    parser.add_argument("--upload", action="store_true", help="If set, upload the produced video to YouTube (requires client_secret.json)")
    args = parser.parse_args()
    run_pipeline(args.keyword, do_upload=args.upload)
