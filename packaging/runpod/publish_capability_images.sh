#!/usr/bin/env bash
# Build and publish the trimmed coderai capability images.
#
#   ./packaging/runpod/publish_capability_images.sh
#
# Interactive and resumable: it skips images already built for this version, so
# re-running after a failure (or a Ctrl-C) picks up where it stopped rather than
# rebuilding 7 GB of torch. Nothing is pushed until every image has been built
# AND has answered /healthz.
#
# Common variations:
#   NS=ghcr.io/someone-else ./publish_capability_images.sh
#   PROFILES="images video" ./publish_capability_images.sh   # just these
#   REBUILD=1 ./publish_capability_images.sh                 # ignore what exists
#   NO_PUSH=1 ./publish_capability_images.sh                 # build only
#   YES=1 ./publish_capability_images.sh                     # no prompts (CI)
set -uo pipefail

cd "$(dirname "$0")/../.."
HERE="packaging/runpod"

NS="${NS:-ghcr.io/nextime}"
REGISTRY="${NS%%/*}"
VERSION="${VERSION:-$(python3 -c 'import re,pathlib; print(re.search(r"__version__ = \"([^\"]+)\"", pathlib.Path("codai/__init__.py").read_text()).group(1))')}"
PROFILES="${PROFILES:-$(ls $HERE/profiles/*.txt | xargs -n1 basename | sed 's/\.txt$//' | grep -v '^core' | grep -v '\.venv-' | tr '\n' ' ')}"
REBUILD="${REBUILD:-0}"
NO_PUSH="${NO_PUSH:-0}"
YES="${YES:-0}"

c_ok=$'\033[32m'; c_warn=$'\033[33m'; c_err=$'\033[31m'; c_off=$'\033[0m'
say(){ printf '%s\n' "$*"; }
ask(){   # ask "question" -> 0 if yes
    [[ "$YES" == "1" ]] && return 0
    read -r -p "$1 [y/N] " a </dev/tty
    [[ "${a,,}" == "y" || "${a,,}" == "yes" ]]
}

count(){ echo $#; }
say "=============================================================="
say " coderai capability images  ->  ${NS}   (version ${VERSION})"
say " profiles: ${PROFILES}"
say "=============================================================="

# ---- preflight ---------------------------------------------------------- #
command -v docker >/dev/null || { say "${c_err}docker not found${c_off}"; exit 1; }
docker info >/dev/null 2>&1 || { say "${c_err}cannot talk to the docker daemon${c_off}"; exit 1; }

# Only BUILDING needs disk. A push uploads what already exists, so a low-space
# check that blocks it is just an outage of its own — which is exactly what it
# did the first time: ten images built, none pushed.
to_build=""
for p in $PROFILES; do
    docker image inspect "${NS}/coderai-${p}:${VERSION}" >/dev/null 2>&1 || to_build="$to_build $p"
done
if [[ -n "${to_build// /}" ]]; then
    dockerroot=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    free_gb=$(df -BG --output=avail "${dockerroot:-/var/lib/docker}" 2>/dev/null | tail -1 | tr -dc '0-9')
    [[ -z "$free_gb" ]] && free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    need=$(( $(count $to_build) * 8 ))
    say "disk: ${free_gb} GB free, roughly ${need} GB needed to build:${to_build}"
    say "      (images share a ~7 GB core layer, so the real cost is much lower"
    say "       after the first one)"
    if (( free_gb < 15 )); then
        say "${c_err}less than 15 GB free — stopping before docker fills the disk${c_off}"
        exit 1
    fi
else
    say "all ${VERSION} images are already built — nothing to build, only push"
fi

# ---- registry login ----------------------------------------------------- #
if [[ "$NO_PUSH" != "1" ]]; then
    if docker login "$REGISTRY" </dev/null >/dev/null 2>&1; then
        say "${c_ok}already logged in to ${REGISTRY}${c_off}"
    else
        say
        say "You need to be logged in to ${REGISTRY} to push."
        if [[ "$REGISTRY" == "ghcr.io" ]]; then
            say "  Create a GitHub token with ${c_warn}write:packages${c_off} at"
            say "    https://github.com/settings/tokens"
            say "  then:  echo \$TOKEN | docker login ghcr.io -u ${NS#*/} --password-stdin"
        fi
        say
        if ask "Log in now?"; then
            docker login "$REGISTRY" </dev/tty || { say "${c_err}login failed${c_off}"; exit 1; }
        else
            say "${c_warn}continuing without login — building only${c_off}"
            NO_PUSH=1
        fi
    fi
fi

# ---- build -------------------------------------------------------------- #
built=(); failed=(); skipped=()
for p in $PROFILES; do
    tag="${NS}/coderai-${p}:${VERSION}"
    if [[ "$REBUILD" != "1" ]] && docker image inspect "$tag" >/dev/null 2>&1; then
        say "${c_ok}skip${c_off}  ${p} — ${tag} already built (REBUILD=1 to force)"
        skipped+=("$tag"); built+=("$tag")
        continue
    fi
    say
    say "---------------- building ${p} ----------------"
    if "$HERE/build_capability_image.sh" "$p" "$tag"; then
        docker tag "$tag" "${NS}/coderai-${p}:latest"
        built+=("$tag")
        say "${c_ok}ok${c_off}    ${p}"
    else
        failed+=("$p")
        say "${c_err}FAIL${c_off}  ${p} — see the output above"
    fi
done

say
say "================== build summary =================="
for t in "${built[@]}"; do
    printf '  %s%-8s%s %-50s %s MB\n' "$c_ok" "built" "$c_off" "$t" \
        "$(( $(docker image inspect "$t" --format '{{.Size}}') / 1000000 ))"
done
for p in "${failed[@]:-}"; do
    [[ -n "$p" ]] && printf '  %s%-8s%s %s\n' "$c_err" "failed" "$c_off" "$p"
done
[[ ${#skipped[@]} -gt 0 ]] && say "  (${#skipped[@]} reused from a previous run)"

if [[ ${#failed[@]} -gt 0 ]]; then
    say
    say "${c_warn}Some profiles failed. Usually a wheel that needs a compiler:${c_off}"
    say "  add the apt packages to ${HERE}/profiles/<profile>.build-deps"
    say "  (installed and purged in the same layer, so they don't ship), then re-run."
    ask "Push the ${#built[@]} image(s) that DID build?" || exit 1
fi
[[ ${#built[@]} -eq 0 ]] && { say "${c_err}nothing built${c_off}"; exit 1; }

# ---- push --------------------------------------------------------------- #
if [[ "$NO_PUSH" == "1" ]]; then
    say
    say "Built, not pushed (NO_PUSH=1 or no login). To push later, re-run this script."
    exit 0
fi

say
say "About to push ${#built[@]} image(s) to ${NS}."
say "${c_warn}These become publicly pullable once you make each package public.${c_off}"
ask "Push now?" || { say "not pushing"; exit 0; }

push_failed=()
for t in "${built[@]}"; do
    say "-- pushing ${t}"
    docker push "$t" || push_failed+=("$t")
    docker push "${t%:*}:latest" || push_failed+=("${t%:*}:latest")
done

say
if [[ ${#push_failed[@]} -gt 0 ]]; then
    say "${c_err}push failures: ${push_failed[*]}${c_off}"
else
    say "${c_ok}all ${#built[@]} images pushed to ${NS}${c_off}"
fi

# ---- sign ------------------------------------------------------------------ #
# cosign signatures next to each pushed digest (packaging/sign-images.sh);
# users verify with packaging/cosign.pub. SIGN=0 skips it.
if [[ "${SIGN:-1}" == "1" && ${#push_failed[@]} -eq 0 ]]; then
    sign_list=()
    for t in "${built[@]}"; do sign_list+=("$t" "${t%:*}:latest"); done
    IMAGES="${sign_list[*]}" NS="$NS" ./packaging/sign-images.sh || say "${c_warn}signing failed — run ./packaging/sign-images.sh all later${c_off}"
fi

# ---- make them public --------------------------------------------------- #
if [[ "$REGISTRY" == "ghcr.io" ]]; then
    owner="${NS#*/}"
    say
    say "${c_warn}GHCR publishes new packages as PRIVATE, and there is no API to change${c_off}"
    say "${c_warn}that${c_off} — the Packages REST API is list/get/delete/restore only, so this is a"
    say "click per package. A RunPod pod cannot pull a private image without a"
    say "registry-credential id, so open each of these and set Visibility → Public:"
    say
    for t in "${built[@]}"; do
        # The package name is the LAST path segment: ghcr.io/<owner>/<name>:<tag>.
        # Stripping only the registry would leave the owner in the name and the
        # URL would 404.
        name="${t##*/}"; name="${name%%:*}"
        say "  https://github.com/users/${owner}/packages/container/${name}/settings"
    done
    say
    say "(Scroll to 'Danger Zone' → 'Change visibility' → Public.)"
fi

# ---- what to do with them ----------------------------------------------- #
say
say "Point a capability at one (config.json):"
say '  "remotes": {'
say '    "endpoints": {"images": "runpod"},'
cat <<EXAMPLE
    "pods": {"images": {"image": "${NS}/coderai-images:${VERSION}",
                        "min_vram_gb": 24, "max_hourly_usd": 0.6,
                        "max_pods": 2, "idle_timeout_s": 300}}
  }
EXAMPLE
say "Several capabilities can share one pod with a common \"pool\" — see"
say "docs/remote-execution.md."
