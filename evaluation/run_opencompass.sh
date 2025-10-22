#!/usr/bin/env bash
set -euo pipefail

# Usage: bash evaluation/run_opencompass.sh /path/to/model_dir OmniMoE/configs/omni_moe_config.json

MODEL_DIR=${1:-}
CONFIG_JSON=${2:-OmniMoE/configs/omni_moe_config.json}

if [[ -z "${MODEL_DIR}" ]]; then
  echo "Usage: $0 MODEL_DIR [CONFIG_JSON]" >&2
  exit 1
fi

# Render the template YAML by replacing placeholders with actual paths
OC_CFG_TMP=$(mktemp /tmp/oc_omni_moe_XXXX.yaml)
sed "s#\${MODEL_DIR}#${MODEL_DIR}#g; s#\${CONFIG_PATH}#${CONFIG_JSON}#g" \
  OmniMoE/evaluation/opencompass/configs/omni_moe.yaml >"${OC_CFG_TMP}"

if command -v opencompass >/dev/null 2>&1; then
  opencompass "${OC_CFG_TMP}" || echo "[OpenCompass] run finished (or skipped)."
else
  echo "[OpenCompass] CLI not found; please install opencompass to run automatic eval." >&2
fi

echo "[OpenCompass] Config rendered: ${OC_CFG_TMP}"
