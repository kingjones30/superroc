# Releasing

Super Roc uses squash merges and Conventional Commit pull request titles to keep
`main` machine-readable. A pull request may contain arbitrary work-in-progress
commits; its title becomes the single commit retained on `main`.

## Pull request titles

Titles have this form:

```text
<type>[(optional-scope)][!]: <description>
```

The accepted types are:

| Type | Purpose | Release effect |
| --- | --- | --- |
| `fix` | Correct user-visible behavior | Patch |
| `perf` | Improve user-visible performance | Patch |
| `feat` | Add backward-compatible functionality | Minor |
| `docs` | Change documentation only | None |
| `test` | Change tests only | None |
| `build` | Change the build system or dependencies | None by default |
| `ci` | Change automation only | None |
| `chore` | Repository maintenance | None |
| `refactor` | Restructure code without changing behavior | None |
| `style` | Change formatting without changing behavior | None |
| `revert` | Revert an earlier change | Determined from the reverted change |

Use a short, lowercase scope when it adds useful ownership context. Examples:

```text
fix(cache): preserve image prefixes after cancellation
perf(qwen): make prompt tokenization linear
feat(server): support constrained JSON output
docs: explain the release policy
feat(server)!: remove the legacy completion endpoint
```

Breaking changes must include `!` before the colon. Do not rely only on a pull
request body footer: the title is the canonical release signal retained by a
single-line squash merge.

## Version policy

Release versions follow Semantic Versioning. The compatibility surface is the
documented command-line interface, HTTP API and public Nix flake outputs. A
change is breaking when an existing documented use must be changed to keep
working.

Before `1.0.0`, Super Roc uses this policy:

- `fix` and `perf` changes increment the patch version.
- `feat` changes increment the minor version.
- breaking changes increment the minor version.
- `1.0.0` is an explicit stability decision and is never selected implicitly.

After `1.0.0`, breaking changes increment the major version. A release uses the
largest required increment among all changes since the previous release.
Ordinary pull requests do not bump the version; the release commit updates the
version and the changelog.

Git tags use `vX.Y.Z`. A released version is immutable. Development builds keep
their exact Git revision so diagnostics can identify the source independently
of the release version.

## Release flow

`version.txt` is the canonical release version. A release is cut by a
maintainer:

1. Run the GPU suite on Strix Halo (`cmake --preset gpu-test`, then
   `ctest -L gpu`) and the CPU suite (`cmake --preset cpu-test`, then
   `ctest -L cpu`) on the commit to release, plus `tools/ci/check-format.py`,
   `tools/ci/check-docs.py` and `tools/ci/check-dependencies.py`.
2. Update `version.txt` and move the `[Unreleased]` section of `CHANGELOG.md`
   under the new version and date in the same commit.
3. Tag that commit `vX.Y.Z` and publish a GitHub Release with the changelog
   section as its notes.
4. Build the release binary from the tag with
   `-DSUPERROC_RELEASE_VERSION=X.Y.Z`, package it with
   `tools/release/package-linux.sh <binary> X.Y.Z <out-dir>` and attach both
   files to the release:

   ```bash
   gh release upload vX.Y.Z \
     <out-dir>/superroc-X.Y.Z-linux-x86_64-rocm7.2-gfx1151.tar.gz \
     <out-dir>/SHA256SUMS
   ```

   The package holds the stripped `superroc` binary, `INSTALL.md`, the license
   and the third-party notices. The script refuses a binary whose `--version`
   does not report `X.Y.Z`.

GitHub Actions runs the CPU contracts and repository checks on every push and
pull request and once a day on `main`. GPU tests need Strix Halo hardware and
run on the maintainers' machines before a release.

Release identity and source identity stay separate. A development build reports
`Super Roc development (<revision>)`; a build configured with
`-DSUPERROC_RELEASE_VERSION=X.Y.Z` from a `vX.Y.Z` tag reports
`Super Roc X.Y.Z (<revision>)`.

A container image for the Lemonade backend is published per release as
`ghcr.io/kingjones30/superroc-lemonade:X.Y.Z`.
