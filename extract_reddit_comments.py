import praw

# Fill in your credentials here
CLIENT_ID = ''
CLIENT_SECRET = ''
USER_AGENT = 'script:PowerBICommentScraper:v1.0 (by /u/Greedy-Band-9626)'

# Initialize Reddit instance
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

# URL of the Reddit post you want to scrape
post_url = 'https://www.reddit.com/r/PowerBI/comments/1lm13jo/what_does_future_looks_like_for_power_bi_in_next/'

# Get submission object
submission = reddit.submission(url=post_url)

# Make sure all comments are loaded
submission.comments.replace_more(limit=None)

# Collect all comments
comments = []
for comment in submission.comments.list():
    comments.append(comment.body)

# Save to a text file or print
with open('BI_comments.txt', 'w', encoding='utf-8') as f:
    for comment in comments:
        f.write(comment + "\n\n")

print(f"Collected {len(comments)} comments!")
