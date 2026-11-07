# RL (GRPO) image: verl 0.8, sglang 0.5.6.post2, torch 2.9.1 (CUDA 12.9), transformers 4.57.3.
#   docker build -t grounded-reasoning-rl envs/rl
#   docker run --gpus all --ipc=host -v $PWD:/workspace -v $GR_DATA:/data -v $GR_RUNS:/runs \
#     -e GR_DATA=/data -e GR_RUNS=/runs -e JUDGE_URL=... grounded-reasoning-rl bash scripts/train_rl.sh grounded 17
FROM verlai/verl:sgl056.latest

# verl as vendored by Thinking with Visual Grounding (unmodified); the base image's Python is the system Python,
# hence --break-system-packages (PEP 668).
ARG TVG_COMMIT=b76028f04e04fe241569b4c029467932bdf92588
RUN git clone https://github.com/Jun-Kai-Zhang/visually_grounded_thinking /opt/visually_grounded_thinking \
    && git -C /opt/visually_grounded_thinking checkout ${TVG_COMMIT} \
    && pip install --break-system-packages --no-deps -e /opt/visually_grounded_thinking/verl \
    && pip install --break-system-packages qwen-vl-utils==0.0.14

WORKDIR /workspace
