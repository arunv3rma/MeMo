

import json
import os
import pickle
from typing import List, Optional, Dict
from datetime import datetime
import random
import base64

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
# for encoding/decoding messages in base64
from base64 import urlsafe_b64decode, urlsafe_b64encode
# for dealing with attachement MIME types
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from mimetypes import guess_type as guess_mime_type
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from cartridges.data.gmail.tools import Message, Thread
from cartridges.data.gmail.utils import authenticate_gmail_api

# SETUP INSTRUCTIONS:
# Follow the steps here to get a credentials.json file: https://thepythoncode.com/article/use-gmail-api-in-python
# Then move it to cartridges/secrets/credentials.json

# Initialize FastMCP server
mcp = FastMCP("weather")

@mcp.tool()
async def list_labels() -> List[str]:
    """List all labels in the user's mailbox."""
    global service
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])
    return [label['name'] for label in labels]



@mcp.tool()
async def fetch_threads(
    num_threads: int = 10,
    sample_random: bool = False,
    label_names: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch email threads from Gmail with flexible filtering and sampling options.

    Parameters
    ----------
    num_threads : int
        Maximum number of threads to return.
    sample_random : bool
        If True, randomly sample ``num_threads`` threads from those that match
        the filters. Otherwise, return the most recent threads.
    label_names : list[str] | None
        Restrict results to threads containing *all* of these label names
        (e.g. ``["INBOX", "STARRED"]``). If omitted, no label filtering is
        applied.
    start_date : str | None
        Inclusive lower-bound on thread date in ``YYYY-MM-DD`` format.
    end_date : str | None
        Inclusive upper-bound on thread date in ``YYYY-MM-DD`` format.

    Returns
    -------
    list[dict]
        Thread metadata dictionaries as returned by the Gmail API.
    """
    global service  # use the authenticated Gmail service created at runtime

    # ------------------------------------------------------------------ #
    # 1. Resolve label names -> label IDs                                #
    # ------------------------------------------------------------------ #
    label_ids: Optional[List[str]] = None
    if label_names:
        label_resp = service.users().labels().list(userId="me").execute()
        name_to_id = {lbl["name"]: lbl["id"] for lbl in label_resp.get("labels", [])}
        label_ids = [name_to_id[name] for name in label_names if name in name_to_id]

        # If none of the supplied label names exist, return early
        if not label_ids:
            return []

    # ------------------------------------------------------------------ #
    # 2. Construct the Gmail search query                                #
    # ------------------------------------------------------------------ #
    query_parts: List[str] = []
    if start_date:
        after_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        query_parts.append(f"after:{after_ts}")
    if end_date:
        # Gmail's `before:` is exclusive, so add a day to make it inclusive
        before_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86_400
        query_parts.append(f"before:{before_ts}")
    query = " ".join(query_parts) if query_parts else None

    # ------------------------------------------------------------------ #
    # 3. Retrieve candidate threads *and* load their messages            #
    # ------------------------------------------------------------------ #
    # The Gmail `threads.list` endpoint returns only a lightweight
    # representation (id + snippet).  To obtain every e-mail in each
    # thread we must:
    #   1. List thread IDs that satisfy the search / label filters.
    #   2. Call `threads.get` for each ID to fetch the full thread object,
    #      whose `messages` field contains all of the e-mails.
    # ------------------------------------------------------------------ #
    target_pool = num_threads * 5 if sample_random else num_threads  # size of candidate set
    thread_ids: List[str] = []
    page_token: Optional[str] = None

    # --- 3a. Collect thread IDs that match the query/label filters --- #
    while len(thread_ids) < target_pool:
        list_resp = (
            service.users()
            .threads()
            .list(
                userId="me",
                q=query,
                labelIds=label_ids,
                maxResults=min(target_pool - len(thread_ids), 100),
                pageToken=page_token,
            )
            .execute()
        )
        thread_ids.extend([t["id"] for t in list_resp.get("threads", [])])
        page_token = list_resp.get("nextPageToken")
        if not page_token:
            break  # exhausted all pages

    if not thread_ids:
        return []

    # --- 3b. Fetch full thread objects (including every e-mail) ------ #
    threads: List[Dict] = []
    for th_id in thread_ids:
        full_thread = (
            service.users()
            .threads()
            .get(userId="me", id=th_id, format="full")
            .execute()
        )
        threads.append(full_thread)
        # Stop early if we've already gathered enough candidates
        if len(threads) >= target_pool:
            break

    # ------------------------------------------------------------------ #
    # 4. Select the final set of threads                                 #
    # ------------------------------------------------------------------ #
    if sample_random:
        threads = random.sample(threads, min(num_threads, len(threads)))
    else:
        threads = threads[:num_threads]

    # Convert the threads to the desired output format
    output = threads
    out = []
    for thread in threads:
        # thread = json.loads(thread.text)

        thread = Thread(
            id=thread["id"],
            # Build Message objects from each Gmail API message, extracting the
            # relevant header fields in a robust way.
            messages=[
                Message(
                    id=msg.get("id"),
                    subject=next(
                        (h["value"] for h in msg.get("payload", {}).get("headers", [])
                         if h["name"].lower() == "subject"),
                        "",
                    ),
                    from_address=next(
                        (h["value"] for h in msg.get("payload", {}).get("headers", [])
                         if h["name"].lower() == "from"),
                        "",
                    ),
                    to_addresses=[
                        addr.strip()
                        for addr in next(
                            (h["value"] for h in msg.get("payload", {}).get("headers", [])
                             if h["name"].lower() == "to"),
                            "",
                        ).split(",")
                        if addr.strip()
                    ],
                    date=next(
                        (h["value"] for h in msg.get("payload", {}).get("headers", [])
                         if h["name"].lower() == "date"),
                        "",
                    ),
                    snippet=msg.get("snippet", ""),
                    # The Gmail API encodes message bodies in base64url. Decode if present.
                    content=(
                        base64.urlsafe_b64decode(msg.get("payload", {}).get("body", {}).get("data", "").encode())
                        .decode(errors="replace")  
                    ),
                    raw=msg
                )
                for msg in thread.get("messages", [])
            ]
        )
        out.append(thread)
    return out 

if __name__ == "__main__":
    global service
    service = authenticate_gmail_api()

    mcp.run(transport="stdio")