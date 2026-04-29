# Remote Publishing Steps

This local repository is already initialized with anonymous git metadata.

## Before You Push

- Create or use an anonymous GitHub account or organization.
- Do not use a personal GitHub username for the submission link.
- Verify that the account profile, avatar, linked organizations, and public email do not identify the authors.

## Publish Commands

From the repository root:

```bash
git remote add origin https://github.com/<anonymous-account-or-org>/<anonymous-repo>.git
git push -u origin main
```

## Optional Final Checks

```bash
git log --format=fuller -1
git remote -v
rg -n "prajvisaxena|FedWell|/Users/|netscratch|@.*\\.edu|@gmail" .
```

The remote should show only the anonymous account or organization, and the latest commit should show the anonymous git identity configured for this repo.
