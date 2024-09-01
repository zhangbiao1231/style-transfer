import logging
import subprocess
import urllib
from pathlib import Path

import requests
import torch
def curl_download(url,filename,*,silent: bool=False) -> bool:
    """Download a file from a url to a filename using curl."""
    silent_option = "sS" if silent else ""
    proc = subprocess.run(
        [
            "curl",
            "-#",
            f"-{silent_option}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ]
    )
    return proc.returncode == 0
def safe_download(file,url,url2=None,min_bytes = 1e0,error_msgs = ""):
    """
    Downloads a file from a URL (or alternate URL) to a specified path if file is above a minimum size.

    Removes incomplete downloads.
    """
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file {file} does not exist or size <min_bytes = {min_bytes}"
    try:
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url,str(file),progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes ,assert_msg #check
    except Exception as e:
        if file.exists():
            file.unlink()
        LOGGER.info(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        # curl download, retry and resume on fail
        curl_download(url2 or url,file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            file.unlink() # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msgs}")
        LOGGER.info("")
def attempt_download(file,repo="ultralytics/yolov5",release=" "):
    """Downloads a file from GitHub release assets or via direct URL if not found locally, supporting backup
        versions.
        """
    from utils.general import LOGGER
    #TODO 此处有检查版本的代码 github_assets
    def github_assets(repository, version="latest"):
        """Fetches GitHub repository release tag and asset names using the GitHub API."""
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v7.0
        response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}").json()  # github api
        return response["tag_name"], [x["name"] for x in response["assets"]]  # tag, assets
    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")): # download
            url = str(file).replace(":/","://")
            file = name.split("?")[0] #parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")
            else:
                safe_download(file=file,url=url,min_bytes=1e5)

        # GitHub assets
        assets = [f"yolov5{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release
        if name in assets:
            file.parent.mkdir(parents=True,exist_ok=True) # make parent dir (if required)
            safe_download(
                file = file,
                url = f"https://github.com/{repo}/releases/download/{tag}/{name}",
                url2=None,
                min_bytes=1e5,
                error_msgs=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}"
            )

    return str(file)




