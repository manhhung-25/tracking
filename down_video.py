import yt_dlp

def download_youtube_video(url):
    # Cấu hình tuỳ chọn tải xuống
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best', # Tải video và audio chất lượng cao nhất
        'merge_output_format': 'mp4',         # Xuất ra định dạng MP4
        'outtmpl': '%(title)s.%(ext)s',       # Đặt tên file là tiêu đề của video
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Đang tiến hành lấy dữ liệu và tải: {url} \nVui lòng đợi trong giây lát...")
            ydl.download([url])
            print("\n🎉 Đã tải video thành công!")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi trong quá trình tải: {e}")

if __name__ == "__main__":
    # Link video bạn đã yêu cầu
    video_url = "https://www.youtube.com/watch?v=d4-udwA0xUo"
    download_youtube_video(video_url)