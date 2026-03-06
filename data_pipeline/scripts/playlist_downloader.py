import yt_dlp
import sys
import os
import concurrent.futures
from yt_dlp.utils import sanitize_filename
import time
def download_video(video_url, ydl_opts):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        ydl.download([video_url])

def download_playlist(playlist_url, max_workers=5):
    output_dir = 'playlist'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Extracting playlist info for: {playlist_url}")
    
    # First, extract the playlist information without downloading
    extract_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
    }
    
    with yt_dlp.YoutubeDL(extract_opts) as ydl: # type: ignore
        info = ydl.extract_info(playlist_url, download=False)
        if not info:
            print("Failed to extract info from playlist")
            return
            
        playlist_title = info.get('title') or 'Unknown_Playlist'
        entries = info.get('entries', [])
        
    if not entries:
        print("No videos found in playlist.")
        return
        
    safe_playlist_title = sanitize_filename(f"{playlist_title}_{str(time.time())}", restricted=True)
    print(f"Found {len(entries)} videos in '{playlist_title}'. Starting parallel downloads...")
    
    # Prepare download tasks
    download_tasks = []
    
    for index, entry in enumerate(entries, start=1):
        if not entry:
            continue
            
        video_url = entry.get('url')
        if not video_url:
            video_url = f"https://www.youtube.com/watch?v={entry.get('id')}"
            
        # Customize ydl_opts for each video to maintain playlist structure
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            # Manually inject playlist title and index
            'outtmpl': f'{output_dir}/{safe_playlist_title}/{index:02d}_%(title)s_%(id)s.%(ext)s',
            'ignoreerrors': True,
            'no_warnings': True,
            'quiet': True, # Keep output clean during parallel downloads
        }
        
        download_tasks.append((video_url, ydl_opts))

    # Execute downloads concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for url, opts in download_tasks:
            futures.append(executor.submit(download_video, url, opts))
            
        # Wait for all futures to complete
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
                print(f"Completed {i}/{len(entries)} videos.")
            except Exception as e:
                print(f"An error occurred: {e}")

    print(f"Finished downloading playlist: {playlist_title}")

if __name__ == '__main__':
    playlists_file = 'playlists.txt'
    if not os.path.exists(playlists_file):
        print(f"Error: '{playlists_file}' not found. Please create it and add playlist URLs, one per line.")
        sys.exit(1)
        
    with open(playlists_file, 'r', encoding='utf-8') as f:
        urls = f.readlines()
        
    for url in urls:
        url = url.strip()
        if url and not url.startswith('#'):
            download_playlist(url)
