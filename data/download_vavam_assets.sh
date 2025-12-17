#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: download_vavam_assets.sh --model <MODEL>

Downloads the tokenizer, detokenizer, and the selected VaVAM weights from the
VideoActionModel v1.0.0 release.

Options:
  --model <MODEL>   Required. One of: VaVAM-S, VaVAM-B, VaVAM-L.
  --list            Print the available model choices and exit.
  -h, --help        Show this help text and exit.
USAGE
}

list_models() {
  cat <<'LIST'
Available models:
  VaVAM-S  (width 768)
  VaVAM-B  (width 1024)
  VaVAM-L  (width 2048, multi-part archive)
LIST
}

ensure_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf 'Error: required command "%s" not found in PATH.\n' "$cmd" >&2
    exit 1
  fi
}

verify_checksum() {
  local file_path="$1"
  local expected="$2"
  local actual=""

  if [[ -z "$expected" ]]; then
    printf 'Error: no checksum registered for %s.\n' "$file_path" >&2
    exit 1
  fi

  if [[ ! -f "$file_path" ]]; then
    printf 'Error: expected file %s not found for checksum verification.\n' "$file_path" >&2
    exit 1
  fi

  actual="$(sha256sum "$file_path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    printf 'Error: checksum mismatch for %s.\nExpected: %s\nActual:   %s\n' "$file_path" "$expected" "$actual" >&2
    rm -f "$file_path"
    exit 1
  fi

  printf '[sha ] %s verified.\n' "$(basename "$file_path")"
}

normalize_model() {
  local input="$1"
  case "$input" in
    VaVAM-S|vavam-s|S|s|small|Small) echo "VaVAM-S" ;;
    VaVAM-B|vavam-b|B|b|base|Base) echo "VaVAM-B" ;;
    VaVAM-L|vavam-l|L|l|large|Large) echo "VaVAM-L" ;;
    *) return 1 ;;
  esac
}

download_file() {
  local url="$1"
  local dest_dir="$2"
  local filename
  filename="${url##*/}"

  if [[ -f "$dest_dir/$filename" ]]; then
    printf '[skip] %s already exists.\n' "$filename"
    return
  fi

  printf '[get ] %s\n' "$filename"
  curl -fL --retry 3 --retry-all-errors --continue-at - \
    --output "$dest_dir/$filename" "$url"
}

main() {
  local model=""

  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        shift
        [[ $# -gt 0 ]] || { printf 'Error: --model requires a value.\n' >&2; exit 1; }
        if ! model="$(normalize_model "$1")"; then
          printf 'Error: unknown model "%s".\n' "$1" >&2
          list_models
          exit 1
        fi
        ;;
      --list)
        list_models
        exit 0
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        printf 'Error: unrecognized argument "%s".\n' "$1" >&2
        usage
        exit 1
        ;;
    esac
    shift
  done

  if [[ -z "$model" ]]; then
    printf 'Error: --model option is required.\n' >&2
    usage
    exit 1
  fi

  ensure_command curl
  ensure_command sha256sum

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  declare -a TOKEN_URLS=(
    "https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VQ_ds16_16384_llamagen_encoder.jit"
    "https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VQ_ds16_16384_llamagen_decoder.jit"
  )

  declare -A VAVAM_URLS=(
    [VaVAM-S]="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_768_pretrained_139k.pt"
    [VaVAM-B]="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_1024_pretrained_139k.pt"
    [VaVAM-L]="https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_aa \
https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ab \
https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ac \
https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0/VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ad"
  )

  declare -A CHECKSUMS=(
    [VQ_ds16_16384_llamagen_encoder.jit]="b5e2c47d04f8ff8a6a47433252aaeefaa8be16be83db6521521a78fefae7e228"
    [VQ_ds16_16384_llamagen_decoder.jit]="1606c10e8dbc6bcc7143a509e55acdf1c53912c996648797556cb436d19d65ee"
    [VAM_width_768_pretrained_139k.pt]="f9a498571e348214d9e256529af41dfccda32e805bab3d7021b65ed09779323f"
    [VAM_width_1024_pretrained_139k.pt]="129e887db30477d458933369964c98b51fdbeb5041eaf8efca4594e2ca56de18"
    [VAM_width_2048_pretrained_139k.pt]="7d617144d54188354b0fc8945e99aa32c6613266aa2d79ac2379f115b3812965"
    [VAM_width_2048_pretrained_139k_chunked.tar.gz.part_aa]="e5d711c6f0d883165a9e2602b085c67a7336ad01783244b69de2197611d9c3ce"
    [VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ab]="7bbf6a4eca7d3adb2e0ec21c412a489b2b7ba2562fc0765236717d8a344205d1"
    [VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ac]="8598877c82782cefa8dc1e52af50bfc9c2ef71e21a1faff56e63b767cc9168e0"
    [VAM_width_2048_pretrained_139k_chunked.tar.gz.part_ad]="634d89a2273c0463d05916995ac875af45953d858b2cd1769a789518963af31d"
    [VAM_width_2048_pretrained_139k_chunked.tar.gz]="717493396f43b45df1842f0dc179ec63d5af1fe64d3030864398dde768f62cd8"
  )

  local target_dir
  target_dir="$script_dir/drivers/vavam"
  mkdir -p "$target_dir"

  printf 'Downloading tokenizer and detokenizer into %s...\n' "$target_dir"
  for url in "${TOKEN_URLS[@]}"; do
    local filename="${url##*/}"
    download_file "$url" "$target_dir"
    verify_checksum "$target_dir/$filename" "${CHECKSUMS[$filename]}"
  done

  printf '\nDownloading %s weights...\n' "$model"
  local urls
  urls="${VAVAM_URLS[$model]}"
  if [[ -z "$urls" ]]; then
    printf 'Error: download URLs for %s not configured.\n' "$model" >&2
    exit 1
  fi

  local -a url_array=()
  read -r -a url_array <<< "$urls"

  for url in "${url_array[@]}"; do
    local filename="${url##*/}"
    download_file "$url" "$target_dir"
    verify_checksum "$target_dir/$filename" "${CHECKSUMS[$filename]}"
  done

  if [[ "$model" == "VaVAM-L" ]]; then
    ensure_command tar
    local combined_name="VAM_width_2048_pretrained_139k_chunked.tar.gz"
    local combined_path="$target_dir/$combined_name"
    if [[ -f "$combined_path" ]]; then
      printf '[skip] %s already exists.\n' "$combined_name"
    else
      printf '[cat ] Combining VaVAM-L chunks into %s...\n' "$combined_name"
      local -a part_files=()
      local part_url
      for part_url in "${url_array[@]}"; do
        part_files+=("$target_dir/${part_url##*/}")
      done

      local missing_part=0
      local part_file
      for part_file in "${part_files[@]}"; do
        if [[ ! -f "$part_file" ]]; then
          printf 'Error: expected chunk %s not found.\n' "$part_file" >&2
          missing_part=1
        fi
      done
      if [[ $missing_part -ne 0 ]]; then
        exit 1
      fi

      cat "${part_files[@]}" > "${combined_path}.tmp"
      mv "${combined_path}.tmp" "$combined_path"
      printf '[done] Combined archive available at %s\n' "$combined_path"
    fi
    verify_checksum "$combined_path" "${CHECKSUMS[$combined_name]}"

    local extract_marker="${combined_path}.extracted"
    if [[ -f "$extract_marker" ]]; then
      printf '[skip] Extraction already completed for %s.\n' "$combined_name"
    else
      printf '[tar ] Extracting %s into %s...\n' "$combined_name" "$target_dir"
      tar -xzf "$combined_path" -C "$target_dir"
      touch "$extract_marker"
      printf '[done] Extraction finished.\n'
    fi
  fi

  printf '\nAll downloads completed. Files saved to %s\n' "$target_dir"
}

main "$@"
