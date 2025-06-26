import urllib.request
import os

url = "https://raw.githubusercontent.com/networkx/networkx/main/examples/graph/lesmis.gml"
save_path = "data/lesmiserables.gml"

os.makedirs("data", exist_ok=True)

print("Downloading lesmiserables.gml...")
urllib.request.urlretrieve(url, save_path)
print("Download complete!")
