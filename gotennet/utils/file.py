import os
import shutil
import urllib
from pathlib import Path
from typing import List

import requests

from gotennet.utils.logging_utils import get_logger

log = get_logger(__name__)

try:
    from tqdm.rich import tqdm as tqdm_rich_progress_bar

    tqdm_rich_available = True
    log.debug(
        "tqdm.rich and rich.progress components are available. Will use for downloads."
    )
except ImportError:
    tqdm_rich_available = False
    log.debug(
        "tqdm.rich or necessary rich.progress components not available. Downloads will be silent."
    )


def download_file(url: str, save_path: str) -> bool:
    """
    Downloads a file from a given URL and saves it to the specified path.
    It handles potential errors, ensures the target directory exists,
    and displays a progress bar using tqdm.rich's default Rich display if available.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path (including filename) where the file should be saved.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    save_path_opened_for_writing = False
    downloaded_size_final = 0

    try:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            log.info(f"Created directory: {directory}")

        log.info(f"Starting download from: {url} to {save_path}")
        response = urllib.request.urlopen(url)

        total_size_str = response.info().get("Content-Length")
        total_size = None
        if total_size_str:
            try:
                parsed_size = int(total_size_str)
                if parsed_size > 0:
                    total_size = parsed_size
                else:
                    log.warning(
                        f"Content-Length is '{total_size_str}', treating as unknown size for progress bar."
                    )
            except ValueError:
                log.warning(
                    f"Could not parse Content-Length header: '{total_size_str}'. Treating as unknown size for progress bar."
                )

        if tqdm_rich_available:
            with open(save_path, "wb") as out_file:
                save_path_opened_for_writing = True
                # Using tqdm.rich with its default Rich display
                # No need to pass 'progress' or 'options' for custom columns
                with (
                    tqdm_rich_progress_bar(
                        total=total_size,  # total=None is handled by tqdm (no percentage/ETA)
                        desc=f"Downloading {os.path.basename(save_path)}",
                        unit="B",  # Unit for progress (Bytes)
                        unit_scale=True,  # Automatically scale to KB, MB, etc.
                        unit_divisor=1024,  # Use 1024 for binary units (KiB, MiB)
                        # leave=True is default, keeps bar after completion
                    ) as pbar
                ):
                    chunk_size = 8192
                    current_downloaded_size = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(
                            len(chunk)
                        )  # Update tqdm progress bar by bytes read
                        current_downloaded_size += len(chunk)
                    downloaded_size_final = current_downloaded_size
                    pbar.refresh()

            if total_size is not None and downloaded_size_final != total_size:
                log.warning(
                    f"Downloaded size {downloaded_size_final} does not match Content-Length {total_size} for {url}. "
                    f"The file might be incomplete or the server reported an incorrect size."
                )

        else:  # tqdm.rich not available, download silently
            log.info(
                f"Downloading {os.path.basename(save_path)} (tqdm.rich not found, progress bar disabled)"
            )
            with open(save_path, "wb") as out_file:
                save_path_opened_for_writing = True
                shutil.copyfileobj(response, out_file)
            if os.path.exists(save_path):
                downloaded_size_final = os.path.getsize(save_path)
                if total_size is not None and downloaded_size_final != total_size:
                    log.warning(
                        f"Downloaded size {downloaded_size_final} (silent download) does not match Content-Length {total_size} for {url}."
                    )

        log.info(f"File downloaded successfully and saved to: {save_path}")
        return True

    except urllib.error.HTTPError as e:
        log.error(f"HTTP Error {e.code} ({e.reason}) while downloading {url}")
    except urllib.error.URLError as e:
        log.error(f"URL Error ({e.reason}) while downloading {url}")
    except OSError as e:
        log.error(
            f"OS Error ({e.errno}: {e.strerror}) while processing {url} for {save_path}"
        )
    except Exception as e:
        log.error(
            f"An unexpected error occurred during download of {url}: {e}", exc_info=True
        )

    if save_path_opened_for_writing and os.path.exists(save_path):
        try:
            log.warning(
                f"Attempting to remove partially downloaded or corrupted file: {save_path}"
            )
            os.remove(save_path)
        except OSError as rm_e:
            log.error(
                f"Could not remove partially downloaded/corrupted file {save_path}: {rm_e}"
            )

    return False


def download_checkpoint(checkpoint_url: str) -> str:
    """
    Downloads a checkpoint file based on the provided identifier.

    Args:
        checkpoint_url (str): The identifier for the checkpoint. Can be a model name
                              (e.g., "QM9_small_homo"), a direct URL, or a local file path.

    Returns:
        str: The local path to the downloaded checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint cannot be found or downloaded.
        ValueError: If the checkpoint name format is invalid or task/parameters are not supported.
        ImportError: If required modules for validation cannot be imported.
    """
    from gotennet.models.tasks import TASK_DICT

    urls_to_try: List[str] = []
    local_filename: str

    # 1. Determine the nature of checkpoint_url_str: Name, URL, or Path
    parts = checkpoint_url.split("_")
    is_potential_name = len(parts) == 3
    is_url = checkpoint_url.startswith(("http://", "https://"))

    # Condition for being a "name": matches pattern, is not a URL, and is not an existing file path
    # (to avoid misinterpreting a local file named 'task_size_label.ckpt' as a downloadable name)
    is_name_style = (
        is_potential_name and not is_url and not os.path.exists(checkpoint_url)
    )

    if is_name_style:
        task, size, label = parts[0], parts[1], parts[2]

        # --- Validation logic for task, size, label (as in previous examples) ---
        # Example (ensure TASK_DICT etc. are properly defined and accessible):
        if task not in TASK_DICT:
            raise ValueError(f"Task {task} is not supported or TASK_DICT not defined.")

        sizes = ["small", "base", "large"]
        if task == "rMD17":
            sizes = ["base"]
        if size not in sizes:
            raise ValueError(f"Size {size} is not supported.")
        if task == "QM9":
            try:
                from gotennet.datamodules.components.qm9 import qm9_target_dict

                label2idx = dict(
                    zip(
                        qm9_target_dict.values(),
                        qm9_target_dict.keys(),
                        strict=False,
                    )
                )
                if label not in label2idx:
                    raise ValueError(
                        f"Label {label} is not valid for QM9 task. Available labels: {list(label2idx.keys())}"
                    )
            except ImportError:
                raise ImportError(
                    "Could not import qm9_target_dict for QM9 task validation."
                )
        # --- End of validation logic ---

        local_filename = (
            f"gotennet_{label}.ckpt"  # Canonical local filename for this name
        )

        # Generate list of URLs to try for this name
        # Primary URL (Hugging Face)
        primary_hf_url = f"https://huggingface.co/sarpaykent/GotenNet/resolve/main/pretrained/{task.lower()}/{size}/{local_filename}"
        urls_to_try.append(primary_hf_url)

        if len(urls_to_try) == 1:  # Only primary was added
            log.info(
                f"Interpreted '{checkpoint_url}' as a model name. Target URL: {urls_to_try[0]}, Local filename: {local_filename}"
            )
        else:
            log.info(
                f"Interpreted '{checkpoint_url}' as a model name. Will try {len(urls_to_try)} URLs. Local filename: {local_filename}"
            )

    elif is_url:
        # It's a direct URL
        urls_to_try.append(checkpoint_url)
        local_filename = os.path.basename(checkpoint_url)
        log.info(
            f"Interpreted '{checkpoint_url}' as a direct URL. Local filename: {local_filename}"
        )
    else:
        # urls_to_try remains empty; we'll only check for this path locally in the ckpt_dir
        log.info(
            f"Interpreted '{checkpoint_url}' as a potential local path identifier. Effective local filename: {local_filename}"
        )
        return ckpt_path

    # 2. Construct local checkpoint path

    home_dir = Path.home()
    default_dir = os.path.join(home_dir, ".gotennet", "checkpoints")
    ckpt_dir = os.environ.get("CHECKPOINT_PATH", default_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, local_filename)

    # 3. Check if file already exists locally and is valid
    if os.path.exists(ckpt_path):
        if os.path.getsize(ckpt_path) > 0:
            log.info(
                f"Using existing checkpoint '{local_filename}' found locally at '{ckpt_path}'."
            )
            return ckpt_path
        else:
            log.warning(
                f"Local checkpoint '{ckpt_path}' exists but is empty. Will attempt to (re-)download if URLs are available."
            )
            try:
                os.remove(ckpt_path)  # Remove empty file
            except OSError as e:
                log.error(f"Could not remove empty local file '{ckpt_path}': {e}")

    # 4. Attempt to download if URLs are available
    if not urls_to_try:
        # This means input was treated as a local path that wasn't found (or was empty),
        # or it was a name for which no URLs were generated (should not happen if name logic is correct).
        raise FileNotFoundError(
            f"Checkpoint '{local_filename}' not found locally at '{ckpt_path}' and no download URLs were specified or derived."
        )

    download_successful = False
    last_error = None

    for i, url_to_attempt in enumerate(urls_to_try):
        log.info(
            f"Attempting download for '{local_filename}' from URL {i + 1}/{len(urls_to_try)}: {url_to_attempt}"
        )
        try:
            # Check URL accessibility
            response = requests.head(url_to_attempt, allow_redirects=True, timeout=10)
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            log.info(f"Remote URL is valid (HTTP Status: {response.status_code}).")

            # Attempt download
            log.warning(
                f"Downloading checkpoint to '{ckpt_path}' from '{url_to_attempt}'."
            )  # Matches original log level
            download_file(url_to_attempt, ckpt_path)

            # Verify download
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError("Local file not found after download attempt.")
            if os.path.getsize(ckpt_path) == 0:
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)  # Clean up empty file
                raise FileNotFoundError("Downloaded file is empty.")

            log.info(
                f"Successfully downloaded '{local_filename}' to '{ckpt_path}' from '{url_to_attempt}'."
            )
            download_successful = True
            break  # Exit loop on successful download

        except requests.exceptions.HTTPError as e:
            log.warning(
                f"Failed to access '{url_to_attempt}' (HTTP status: {e.response.status_code})."
            )
            last_error = e
        except (
            requests.exceptions.RequestException
        ) as e:  # Catches DNS errors, connection timeouts, etc.
            log.warning(f"Connection error for '{url_to_attempt}': {e}.")
            last_error = e
        except (
            FileNotFoundError
        ) as e:  # From our own post-download checks or if download_file raises it
            log.warning(f"Download or verification failed for '{url_to_attempt}': {e}.")
            last_error = e
            if (
                os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) == 0
            ):  # Clean up if an empty file was created
                try:
                    os.remove(ckpt_path)
                except OSError:
                    pass
        except (
            Exception
        ) as e:  # Catch other errors from download_file or unexpected issues
            log.warning(
                f"An unexpected error occurred during download from '{url_to_attempt}': {e}"
            )
            last_error = e
            if os.path.exists(ckpt_path):  # Clean up potentially corrupt file
                try:
                    os.remove(ckpt_path)
                except OSError:
                    pass

        if i < len(urls_to_try) - 1:  # If there are more URLs to try
            log.info("Trying next available URL...")

    if not download_successful:
        error_message = f"Failed to download checkpoint '{local_filename}' from all provided sources."
        if urls_to_try:
            error_message += f" Tried: {', '.join(urls_to_try)}."
        if last_error:
            log.error(f"{error_message} Last error: {last_error}")
            raise FileNotFoundError(error_message) from last_error
        else:
            log.error(error_message)
            raise FileNotFoundError(error_message)

    return ckpt_path
