import os
import json
import time
import requests
from tqdm import tqdm
from datetime import datetime

# Constants
GITHUB_API_URL = "https://api.github.com"
REPO_OWNER = "pytorch"
REPO_NAME = "pytorch"
ACCESS_TOKEN = json.load(open("access_tokens.json"))["GHB_PAT_2"]

def get_headers():
    headers = {
        "Accept": "application/vnd.github.v3+json"  # Ensure we're using the correct API version
    }
    if ACCESS_TOKEN:
        headers["Authorization"] = f"token {ACCESS_TOKEN}"
    return headers

def get_closed_pull_requests(owner, repo):
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls"
    headers = get_headers()
    params = {
        "state": "closed",
        "per_page": 100  # Fetch up to 100 PRs per page
    }
    response = requests.get(url, headers=headers, params=params)
    handle_rate_limit(response.headers)
    if response.status_code == 404:
        raise Exception(f"Repository '{owner}/{repo}' not found.")
    response.raise_for_status()
    return response.json()

def get_review_comments(owner, repo, pull_number):
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    headers = get_headers()
    response = requests.get(url, headers=headers)
    handle_rate_limit(response.headers)
    if response.status_code == 404:
        raise Exception(f"Pull request #{pull_number} not found in repository '{owner}/{repo}'.")
    response.raise_for_status()
    return response.json()

def handle_rate_limit(headers):
    limit = int(headers['X-RateLimit-Limit'])
    remaining = int(headers['X-RateLimit-Remaining'])
    reset_time = int(headers['X-RateLimit-Reset'])
    current_time = int(time.time())
    
    if remaining == 0:
        sleep_time = reset_time - current_time + 10  # Add some buffer time
        print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)

def main(REPO_OWNER, REPO_NAME):
    current_time = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    DATA_FOLDER = os.path.join("data", "review_comments", REPO_OWNER, REPO_NAME)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    dump_path = os.path.join(DATA_FOLDER, f"{current_time}.jsonl")
    pbar = tqdm() # progress bar.
    try:
        closed_pulls = get_closed_pull_requests(REPO_OWNER, REPO_NAME)
    except Exception as e:
        print(f"Error fetching closed pull requests: {e}")
        return

    for pr in closed_pulls:
        pr_number = pr["number"]
        # pr_title = pr["title"]
        # print(f"Pull Request #{pr_number}: {pr_title}")
        try:
            review_comments = get_review_comments(REPO_OWNER, REPO_NAME, pr_number)
        except Exception as e:
            print(f"Error fetching review comments for PR #{pr_number}: {e}")
            continue

        if not review_comments:
            # print("No review comments.")
            continue
        
        for comment in review_comments:
            # print(comment.keys())
            comment["PR"] = {
                "title": pr['title'], 
                'number': pr['number'],
                "id": pr["id"],
                # "html_url": pr["html_url"],
                # "diff_url": pr["diff_url"],
                # "patch_url": pr["patch_url"],
                # "issue_url": pr["issue_url"],
            }
            comment["user"] = {
                "login": comment["user"]["login"],
                "id": comment["user"]["id"],
                "url": comment["user"]["url"],
                "html_url": comment["user"]["html_url"],
            }
            # user = comment["user"]["login"]
            # body = comment["body"]
            # path = comment["path"]
            # position = comment["position"]
            # created_at = comment["created_at"]
            with open(dump_path, "a") as f:
                f.write(json.dumps(comment)+"\n")
                pbar.update(1)
            # print(f"- Comment by {user} on {created_at} at {path} (position {position}): {body}")

if __name__ == "__main__":
    main()
