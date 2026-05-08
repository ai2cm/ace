# pre-review.md

A PR (corresponding to the current checked-out branch, or one the user has told you) is ready for review.
That review is going to be performed by another team member, but that member has limited time.
You are tasked with performing a "pre-review" to make this PR as easy to review as possible.

For example, this could mean:
 - Re-structuring the PR, either by splitting it into multiple smaller PRs or by re-ordering commits, to make it easier to review.
 - Catching and fixing any issues you can find before the PR is seen by the reviewer.
 - Making comments on the PR to point out potential issues or areas of concern for the reviewer to focus on.
 - Improving the PR description and title.

Focus specifically on making the reviewer's job easier, not on making the PR perfect.
Issues involving readability might be particularly important, especially if the changes are hard to understand or the reason for them is not immediately clear.

Begin by taking a look at the PR, as well as its description and title, and considering areas for improvement.
Communicate with the user about whether:
 - The PR has significant structural issues that should be fixed with significant refactors of the PR.
 - The PR should be significantly restructured, for example by restructuring the commits to have a more logical flow or by splitting into multiple smaller PRs.
 - The PR structure is acceptable, but there are specific issues that should be fixed before review.
 - The PR is in good shape, perhaps you have some comments to make on the PR or updates to its title/description but the code is ready for the reviewer.

When leaving comments on a PR, always start your comment with "Claude: ".
