import json
from github import Github

# Replace with your GitHub token
token = json.load(open("access_tokens.json"))["GHB_PAT_1"]

# Authenticate using your GitHub token
g = Github(token)

# Replace 'owner' and 'repo' with the repository's owner and name
owner = "django"
repo_name = "django"
repo = g.get_repo(f"{owner}/{repo_name}")

# Define the search query
phrase = "best practices"
query = f'repo:{owner}/{repo_name} "{phrase}" in:comments type:pr'

# Search for PRs
search_results = g.search_issues(query, type='pr')

# Fetch PR numbers from the search results
pr_numbers = [issue.number for issue in search_results]

# Dictionary to hold PR data
pr_data = {}

# Extract and collect discussion threads
for pr_number in pr_numbers:
    pr = repo.get_pull(pr_number)
    comments = pr.get_issue_comments()  # Get issue comments (including the initial PR comment)
    review_comments = pr.get_review_comments()  # Get review comments

    # Initialize dictionary for this PR
    pr_data[pr_number] = {
        "title": pr.title,
        "discussion_thread": []
    }

    # Add issue comments to the PR dictionary
    for comment in comments:
        pr_data[pr_number]["discussion_thread"].append({
            "user": comment.user.login,
            "comment": comment.body,
            "type": "issue_comment",
            "created_at": comment.created_at.isoformat()
        })

    # Add review comments to the PR dictionary
    for review_comment in review_comments:
        pr_data[pr_number]["discussion_thread"].append({
            "user": review_comment.user.login,
            "comment": review_comment.body,
            "type": "review_comment",
            "created_at": review_comment.created_at.isoformat()
        })

# Save PR data to a JSON file
with open('pr_comments.json', 'w') as json_file:
    json.dump(pr_data, json_file, indent=4)

print("PR comments saved to pr_comments.json")
