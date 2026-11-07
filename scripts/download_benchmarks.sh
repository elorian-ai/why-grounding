#!/usr/bin/env bash
# Download the 11 benchmarks at pinned revisions into $GR_DATA/benchmarks/ (~15 GB) and write the images the item
# lists point at.   bash scripts/download_benchmarks.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=${GR_DATA:-$REPO/data}/benchmarks
mkdir -p "$D"
# Retried with a growing pause: hf download resumes, and repos of many single images can hit the Hub rate limit.
dl() {
  local attempt
  for attempt in 1 2 3 4 5; do
    hf download "$1" --repo-type dataset --revision "$2" --local-dir "$D/$3" "${@:4}" && return 0
    echo "hf download $1 failed (attempt $attempt of 5), retrying in $((attempt * 60)) s" >&2
    sleep $((attempt * 60))
  done
  return 1
}

dl liuziyan/SpatialMQA 2c297135743209b91fe0729c033b49bb1d72f788 spatialmqa
dl etri-vilab/MultihopSpatial 6c962f5227acc133d2196cbf89f09441c7ccb703 multihopspatial --include "data/*"
dl Phineas476/EmbSpatial-Bench e2733cacf0f20371518c687176e881d79644e61e embspatial --include embspatial_bench.json
# VSR captions; grounded_reasoning.benchmarks fetches their COCO 2017 images from the links in test.jsonl.
dl cambridgeltl/vsr_zeroshot 148b3777ceef4a1bfe46377614980836db7d12f2 vsr --include test.jsonl
dl qizekun/OmniSpatial 6691f3288bb1ff207d6ead4d841b505de08a6fd8 omnispatial --include OmniSpatial-test.zip
unzip -q -o "$D/omnispatial/OmniSpatial-test.zip" -d "$D/omnispatial/unzipped"
# ERQA ships one tfrecord (data/erqa.tfrecord) in its GitHub repository.
[ -d "$D/erqa/ERQA/.git" ] || git clone https://github.com/embodiedreasoning/ERQA "$D/erqa/ERQA"
git -C "$D/erqa/ERQA" checkout -q 5bba488252c5b345a43335bfb9239b6bf12bd91c
dl craigwu/vstar_bench d9ae62c903da0c98336e85c5ee89cd863b04b4da vstar
dl yifanzhang114/MME-RealWorld-lite-lmms-eval f6b0dc81ba4d3c39bd9b4e544578198d365ac084 mmerw
dl HaochenWang/TreeBench a4eb437ab71d14f6ab1a4cf05213d0f72f831b71 treebench
dl Jayant-Sravan/CountQA f92cc6fe46542c61e2916e3d2ae9a911e2216b1a countqa
dl isentropic/FSC147 3e420cb6537e803dd6d4516623ce82a79c0317b8 fsc147
echo "benchmark sources in $D"
"${PYTHON:-python}" -m grounded_reasoning.benchmarks
