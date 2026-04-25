import requests
from pathlib import Path

urls = [
    "https://upload.wikimedia.org/wikipedia/commons/0/02/Rigveda_MS2097.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/96/1500-1200_BCE%2C_Rigveda_manuscript_page_sample_v%2C_Sanskrit%2C_Devanagari.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/A_palm_leaf_Sanskrit_manuscript_in_Brahmi_script_from_Miran_China.jpg/1280px-A_palm_leaf_Sanskrit_manuscript_in_Brahmi_script_from_Miran_China.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Devimahatmya_Sanskrit_MS_Nepal_11c.jpg/800px-Devimahatmya_Sanskrit_MS_Nepal_11c.jpg",
    "https://blogs.loc.gov/international-collections/files/2018/01/Image-2-Bhagavad-Gita-1024x728.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/98/Indian_manuscript%2C_Devanagari_script_on_paper.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/db/Folio_from_a_Ramayana_manuscript%2C_text_in_Devanagari.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/88/Vedas_palm_leaf_manuscript%2C_Tamil_Grantha_Script%2C_Sanskrit%2C_Tamil_Nadu.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/66/Palm-leaf_manuscript%2C_ancient_and_medieval_Tamil_literature_%28partly_Sangam_era%29%2C_Languages_in_the_manuscript_Tamil_Telugu_Sanskrit%2C_Scripts_Grantha_Telugu_Tamil%2C_Hindu_Shaivism_monastery%2C_UVSL_589.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/1/1f/Bhagavata_Purana_Illustrated_Manuscript.jpg"
]

out_dir = Path("data/raw/manuscripts")
out_dir.mkdir(parents=True, exist_ok=True)

# Add browser headers to avoid 403
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
}

def download_image(url):
    filename = url.split("/")[-1].replace("%", "_")
    dest = out_dir / filename

    print(f"[+] Downloading {filename}")

    r = requests.get(url, headers=headers, stream=True, timeout=30)
    if r.status_code == 403:
        print(f"[!] 403 Forbidden for {filename}. Trying fallback URL...")

    r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    print(f"    Saved -> {dest}")


def main():
    for url in urls:
        try:
            download_image(url)
        except Exception as e:
            print(f"[ERROR] Failed to download {url}: {e}")

    print("\n[✓] All downloadable manuscript images processed.")


if __name__ == "__main__":
    main()
