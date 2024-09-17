import pkg_resources
from importlib.metadata import distributions


for dist in distributions():
    metadata = dist.metadata
    if 'Name' not in metadata:
        print(f"Package {dist._path} thiếu trường 'Name' trong metadata.")

