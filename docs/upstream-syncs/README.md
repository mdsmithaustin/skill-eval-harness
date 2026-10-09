# Sync upstream into the fork

Use this procedure to sync `adewale/skill-eval-harness` into
`mdsmithaustin/skill-eval-harness` through a reviewed PR to the fork's `main`.
Keep a dated record of each sync and an annotated checkpoint tag after acceptance.

## Prepare a branch from fork main

1. Start with a clean checkout of the fork. Verify that `origin` names
   `https://github.com/mdsmithaustin/skill-eval-harness.git` with `git remote -v`.
2. Check for the parent remote with `git remote get-url upstream`. If the remote
   does not exist, add it.

   ```sh
   git remote add upstream https://github.com/adewale/skill-eval-harness.git
   ```

   Verify that `upstream` names that URL with `git remote -v` before fetching.
3. Fetch both repositories.

   ```sh
   git fetch origin
   git fetch upstream
   ```

4. Set `SYNC_DATE` to the record date in `YYYY-MM-DD` format. Pin both inputs before
   merging, and create the branch from the fetched fork `main`.

   ```sh
   SYNC_DATE=YYYY-MM-DD
   PREVIOUS_FORK=$(git rev-parse origin/main)
   UPSTREAM_TIP=$(git rev-parse upstream/main)
   git switch -c "sync-upstream-$SYNC_DATE" "$PREVIOUS_FORK"
   git rev-list --left-right --count "$PREVIOUS_FORK...$UPSTREAM_TIP"
   git merge --no-ff --no-commit "$UPSTREAM_TIP"
   ```

   The two counts show fork-only and upstream-only commits before the merge.
   A normal merge retains both histories. Do not replace the sync with a squash or
   a replay of upstream commits.

## Resolve and verify the merge

1. Inventory the fork-only behavior before resolving conflicts. Record which
   upstream implementation owns each shared pipeline and which fork behavior remains.
2. Resolve conflicts at those owners. Keep unrelated local files and lockfiles out
   of the sync. Do not force-push `main`.
3. Run the current [contributor checks](../../CONTRIBUTING.md) and checks for the
   preserved behavior. Record actual commands, outcomes, and skipped checks.
   A nonzero exit or an empty test collection fails verification.
4. Create `docs/upstream-syncs/YYYY-MM-DD.md` using the
   [2026-10-05 record](2026-10-05.md) as the reference shape. Include the full input
   SHAs, merge parents, preservation choices, evidence, and unresolved limits.
   Separate the previous-fork-to-result diff from the upstream-to-result diff.
5. Add the record to the index below and the [canonical docs index](../README.md).
   Commit the normal merge after the conflicts and local checks are resolved.
   Record the resulting merge SHA in the dated record before opening the PR.

## Review and land in the fork

1. Push the sync branch to `mdsmithaustin/skill-eval-harness` and open a PR against
   its `main`. Link the dated record and both comparisons in the PR description.
2. Obtain review of the preservation choices and documentation. Wait for every
   applicable [CI job](../../.github/workflows/ci.yml) to pass on the final PR head.
   Local checks do not replace CI.
3. If review or CI requires repairs, update the branch and the record. Rerun the
   affected checks and wait for CI on the repaired head.
4. Land with a merge commit so the sync merge remains in `main`'s ancestry.
   Do not squash or rebase the PR. Verify that both pinned inputs and the sync
   merge are ancestors of the landed commit. An ancestry check fails with a
   nonzero exit.

   ```sh
   git fetch origin
   git merge-base --is-ancestor "$PREVIOUS_FORK" origin/main
   git merge-base --is-ancestor "$UPSTREAM_TIP" origin/main
   git merge-base --is-ancestor "$SYNC_MERGE" origin/main
   ```

   Set `SYNC_MERGE` to the full recorded merge SHA before these checks.

5. Keep the dated record current as review and CI expose repairs or limits.
   Identify the accepted commit through the PR and fork history. Retain the
   original merge identity even when a follow-up correction becomes the checkpoint.

## Create the accepted checkpoint

1. Select the exact accepted landed commit after review. Wait for green CI on that
   commit, then set `CHECKPOINT_COMMIT` to its full SHA instead of a moving branch name.
   For a retrospective record, select the validated follow-up commit and retain
   the original sync merge SHA in the record.
2. Check that `sync-upstream-YYYY-MM-DD` does not already exist locally or on the
   fork. Do not move an existing checkpoint tag.
3. Create and push the annotated tag to the fork. Include the pinned inputs, sync
   merge, and dated record in its annotation.

   ```sh
   git tag -a "sync-upstream-$SYNC_DATE" "$CHECKPOINT_COMMIT" \
   -m "Upstream sync $SYNC_DATE" \
   -m "Previous fork $PREVIOUS_FORK. Upstream $UPSTREAM_TIP. Sync merge $SYNC_MERGE. Record docs/upstream-syncs/$SYNC_DATE.md."
   git push origin "refs/tags/sync-upstream-$SYNC_DATE"
   ```

4. Verify the remote tag's annotation and peeled commit against
   `CHECKPOINT_COMMIT`. A missing annotation or a different commit fails the check.
   The annotation and fork history identify the checkpoint without a documentation
   update that refers to its own commit.

The tag records an accepted checkpoint. Creating the tag does not approve code.
The `sync-upstream-YYYY-MM-DD` namespace is distinct from version tags such as
`v0.6.0`. Do not create a GitHub release for a sync.

## Dated records

| Date | Record |
| --- | --- |
| 2026-10-09 | [Merge of pinned upstream main with fork behavior preserved](2026-10-09.md) |
| 2026-10-05 | [Merge of upstream main with fork behavior preserved](2026-10-05.md) |
