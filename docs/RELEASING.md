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
3. Tag that commit `vX.Y.Z`, push the tag and publish the GitHub Release with
   the changelog section as its notes
   (`gh release create vX.Y.Z --verify-tag --notes-file <notes>`).
4. The `vX.Y.Z` tag starts `.github/workflows/binaries.yml`, which builds the
   release binary from the tag with `tools/release/build-linux.sh`, packages it
   with `tools/release/package-linux.sh` and attaches
   `superroc-X.Y.Z-linux-x86_64-rocm7.2-gfx1151.tar.gz` and `SHA256SUMS` to the
   release (if the release does not exist yet, the workflow creates it from the
   changelog section). Download the archive on Strix Halo and check that it
   serves a model before announcing the release.

## Nightly builds

The same workflow runs every night. When `main` has changed since the last
nightly build, it builds `main`, moves the `nightly` tag to that commit and
replaces the archive of the `nightly` pre-release. A nightly binary reports
`Super Roc development (<revision>)`. Nightly builds pass the hosted CPU
checks; only tagged releases are tested on Strix Halo before they are
published.

A package can be built on any Ubuntu 24.04 machine, without a GPU:

```bash
sudo tools/release/install-build-deps-ubuntu.sh
tools/release/build-linux.sh . ../build ../stage X.Y.Z
tools/release/package-linux.sh ../stage ../out X.Y.Z
```

The archive holds the stripped `superroc` binary, its data files under `share/`,
`INSTALL.md`, the license and the third-party notices. `package-linux.sh`
refuses a binary whose `--version` does not report the expected version, and
`build-linux.sh` refuses a binary that still names its source or build
directory.

GitHub Actions runs the CPU contracts and repository checks on every push and
pull request and once a day on `main`. GPU tests need Strix Halo hardware and
run on the maintainers' machines before a release.

Release identity and source identity stay separate. A development build reports
`Super Roc development (<revision>)`; a build configured with
`-DSUPERROC_RELEASE_VERSION=X.Y.Z` from a `vX.Y.Z` tag reports
`Super Roc X.Y.Z (<revision>)`.

A container image for the Lemonade backend is published per release as
`ghcr.io/kingjones30/superroc-lemonade:X.Y.Z`.
