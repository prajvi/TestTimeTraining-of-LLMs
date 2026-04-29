# Track Requirements Alignment

This artifact was prepared to match the NeurIPS 2026 Evaluations and Datasets track expectations for anonymous supplementary code.

## Design Goals

The repository is organized so that the linked code artifact is:

- anonymous by construction
- documented and runnable
- lightweight enough for reviewer access
- separated from personal working notes and local infrastructure

## What Was Removed

To preserve double-blind review, the anonymous repo excludes:

- local user paths
- cluster-specific launcher scripts with institution-specific paths
- local token files and token-loading workflows tied to a repo-local secret
- private planning notes and experiment logs
- large intermediate caches and nonessential generated outputs

## What Was Kept

- the executable BSTTT code
- benchmark evaluation entry points
- configs used by the experiments
- manuscript sources and figure assets
- minimal utility scripts for summarization and table generation

## Remote Hosting Note

If this repository is pushed to a public remote for review, the remote itself must not reveal author identity.
A normal personal GitHub account is not anonymous.
Use an anonymous organization/account or an anonymized hosting workflow before sharing the link with reviewers.
