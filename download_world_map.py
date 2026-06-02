import urllib.request

req = urllib.request.Request(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Whole_world_-_land_and_oceans_12000.jpg/1280px-Whole_world_-_land_and_oceans_12000.jpg",
    headers={"User-Agent": "Mozilla/5.0"}
)
with urllib.request.urlopen(req) as r, open("world_map.jpg", "wb") as f:
    f.write(r.read())
