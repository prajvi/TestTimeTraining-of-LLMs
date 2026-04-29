# Anonymity Checklist

Use this checklist before creating or sharing a remote repository for submission.

## Repository Content

- No local absolute paths remain in tracked files.
- No local token files are present.
- No personal notes, planning docs, or cluster-specific paths are present.
- No data caches or large private outputs are present unless they are explicitly needed for review.

## Git Metadata

- Initialize the repository fresh rather than preserving history from the working repo.
- Configure the commit author to an anonymous identity before the first commit.
- Do not push commits authored with a personal name or personal email.

Example:

```bash
git config user.name "Anonymous Authors"
git config user.email "anonymous@example.com"
```

## Remote Hosting

- Do not publish under a personal GitHub username.
- Use an anonymous GitHub organization/account or another anonymous artifact-hosting workflow.
- Check the repository owner, profile, avatar, and linked organizations before sharing the URL.

## Final Scan

- Run a final grep for local usernames, home directories, institution names, and emails.
- Inspect the repository in a logged-out browser session before sharing it.
