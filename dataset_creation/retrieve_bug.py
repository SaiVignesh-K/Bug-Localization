


# from github import Github
# import time

# def correlate_issues_with_pulls(repository_name, access_token):
#     # Initialize GitHub instance with access token
#     g = Github(access_token)

#     # Get the repository
#     repo = g.get_repo(repository_name)

#     # Dictionary to store correlated issues and pull requests
#     correlated_data = {}

#     # Iterate through all pull requests in the repository
#     for pull in repo.get_pulls(state='all'):
#         # Check if the pull request is merged
#         if pull.is_merged():
#             # Extract issue URL from the pull request body
#             issue_url = None
#             if pull.body:
#                 for line in pull.body.split('\n'):
#                     if line.startswith(('Fixes', 'Resolves')):
#                         issue_url = line.split()[-1]
#                         break

#             if issue_url:
#                 # Extract issue ID from the issue URL
#                 issue_id = ''.join(filter(str.isdigit, issue_url))

#                 if issue_id:
#                     try:
#                         # Get information about the issue
#                         issue = repo.get_issue(int(issue_id))

#                         # Store correlated data
#                         correlated_data[issue.title] = {
#                             'issue_id': issue_id,
#                             'pull_request_number': pull.number,
#                             'files_changed': [file.filename for file in pull.get_files()],
#                             'lines_changed': pull.additions + pull.deletions
#                         }
#                         # Print correlated data
#                         print(f"Issue: {issue.title}")
#                         print(f"Issue ID: {issue_id}")
#                         print(f"Pull Request Number: {pull.number}")
#                         print(f"Files Changed: {[file.filename for file in pull.get_files()]}")
#                         print(f"Lines Changed: {pull.additions + pull.deletions}")
#                     except Exception as e:
#                         print(f"Error fetching issue {issue_id}: {e}")

#             # Add a delay of 1 second to avoid hitting rate limits
#             time.sleep(1)

#     return correlated_data

# def save_to_file(correlated_data, filename):
#     with open(filename, 'w') as f:
#         f.write("Issue Title\tIssue ID\tPull Request Number\tFiles Changed\tLines Changed\n")
#         for issue_title, data in correlated_data.items():
#             files_changed = ', '.join(data['files_changed'])
#             f.write(f"{issue_title}\t{data['issue_id']}\t{data['pull_request_number']}\t{files_changed}\t{data['lines_changed']}\n")

# # Example usage
# repository_name = "HouariZegai/Calculator"
# access_token = "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"

# correlated_data = correlate_issues_with_pulls(repository_name, access_token)

# save_to_file(correlated_data, "bug_report.txt")


import csv
from github import Github
import time
import re

def get_issue_number(text):
    # Regular expression to find issue numbers
    pattern = r'#(\d+)'
    matches = re.findall(pattern, text)
    return matches

def correlate_issues_with_pulls(repository_name, access_token):
    # Initialize GitHub instance with access token
    g = Github(access_token)

    # Get the repository
    repo = g.get_repo(repository_name)

    # List to store correlated data
    correlated_data = []

    # Iterate through all pull requests in the repository
    for pull in repo.get_pulls(state='all'):
        # Check if the pull request is merged
        if pull.is_merged():
            # Extract issue numbers from both title and body
            issue_numbers = get_issue_number(pull.title)
            print(issue_numbers)
            if pull.body:
                issue_numbers.extend(get_issue_number(pull.body))

            for issue_number in issue_numbers:
                try:
                    # Get information about the issue
                    issue = repo.get_issue(int(issue_number))

                    # Store correlated data
                    correlated_data.append({
                        'Issue Title': issue.title,
                        'Issue ID': issue_number,
                        'Pull Request Number': pull.number,
                        'Files Changed': ', '.join([file.filename for file in pull.get_files()]),
                        'Lines Changed': pull.additions + pull.deletions
                    })

                    # Print correlated data
                    print(f"Issue: {issue.title}")
                    print(f"Issue ID: {issue_number}")
                    print(f"Pull Request Number: {pull.number}")
                    print(f"Files Changed: {[file.filename for file in pull.get_files()]}")
                    print(f"Lines Changed: {pull.additions + pull.deletions}")
                except Exception as e:
                    print(f"Error fetching issue {issue_number}: {e}")

            # Add a delay of 1 second to avoid hitting rate limits
            time.sleep(10)

    return correlated_data

def save_to_csv(correlated_data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Issue Title', 'Issue ID', 'Pull Request Number', 'Files Changed', 'Lines Changed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in correlated_data:
            writer.writerow(data)

# Example usage
repository_name = "netflix/hollow"
access_token = "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"

correlated_data = correlate_issues_with_pulls(repository_name, access_token)

save_to_csv(correlated_data, "netflix_bug_report.csv")

