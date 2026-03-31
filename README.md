# wyoming-multi-tts for Home Assistant

Multi-engine TTS server for Home Assistant built on Wyoming Protocol, with a web control panel on port `8280`.

## What It Does

- Exposes one Wyoming TTS endpoint on port `10210`
- Lets you switch the active engine from the web UI
- Loads engines manually from the UI and remembers the last loaded engine across restarts
- Unloads the previous engine when switching to save VRAM
- Uses GPU first and falls back to CPU when needed
- Tracks synthesis timing for quick comparisons

## Included Engines

- `Chatterbox`
- `WhisperSpeech`
- `XTTS-v2`
- `Qwen3-TTS`
- `MMS TTS Polish`
- `Fish Audio S2 Pro`

## Requirements

- Linux host with Docker and Docker Compose
- NVIDIA GPU recommended
- CPU-only mode also works as a fallback, but it is much slower
- NVIDIA Container Toolkit installed on the host
- CUDA 12.6 compatible driver on the host
- Hugging Face token in `.env`
- External Docker network named `bridge-network`, or another name passed as `DOCKER_NETWORK`

## Quick Start

1. Copy the environment template:

```bash
cp .env.example .env
```

2. Put your Hugging Face token in `.env`:

```env
HF_TOKEN=your_huggingface_token
DOCKER_NETWORK=bridge-network
```

3. Create the external Docker network once:

```bash
docker network create bridge-network
```

4. Build and start the service:

```bash
docker compose up -d --build
```

5. Open the UI:

- Web UI: `http://YOUR_HOST:8280`
- Wyoming TTS: `tcp://YOUR_HOST:10210`

## Persistent Data

The compose file uses project-relative bind mounts, so the following folders are preserved between rebuilds:

- `./data/hf-cache`
- `./data/state`
- `./data/speakers`

## How To Use

1. Open the web UI on port `8280`
2. Click `Load` on the engine you want to use
3. Wait for `ready`
4. Optionally test synthesis from the UI
5. Add the Wyoming integration in Home Assistant and point it to port `10210`

The active engine and last successful loaded state are persisted. After a container restart, the same engine is auto-loaded again.

## Rebuild Notes

- Normal rebuild:

```bash
docker compose up -d --build
```

- Full test suite:

```bash
./.venv/bin/pytest -q
```

- Local syntax check:

```bash
python3 -m compileall app tests
```

## Optional: Prebuild `flash-attn` Wheel

`Qwen3-TTS` can reuse a local prebuilt `flash-attn` wheel to avoid a long source build.

```bash
./scripts/build_flash_attn_wheel.sh
```

This writes a compatible wheel into `wheelhouse/`. Docker will reuse it automatically on the next build.

## Notes For Another Computer

To run this project on another machine, make sure all of the following are true:

- `.env` exists and contains `HF_TOKEN`
- the external Docker network exists
- NVIDIA Container Toolkit is installed if you want GPU mode
- ports `8280` and `10210` are available
- the host GPU driver is compatible with CUDA 12.6

If no usable GPU is available, engines may still fall back to CPU, but latency will be much worse and some engines may be impractical.

## License

This repository's code is released under the MIT license. See [LICENSE](./LICENSE).

Downloaded models and third-party runtimes keep their own licenses. Important ones used by this project include:

- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`: Apache-2.0  
  https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
- `ResembleAI/chatterbox`: MIT  
  https://huggingface.co/ResembleAI/chatterbox
- `facebook/mms-tts-pol`: CC-BY-NC-4.0  
  https://huggingface.co/facebook/mms-tts-pol
- `coqui/XTTS-v2`: Coqui Public Model License  
  https://huggingface.co/coqui/XTTS-v2
- `fishaudio/s2-pro`: check the upstream model page before commercial use  
  https://huggingface.co/fishaudio/s2-pro
- `WhisperSpeech`: check the upstream project and model artifacts you deploy  
  https://github.com/WhisperSpeech/WhisperSpeech

Because the engine set mixes permissive and restricted model licenses, commercial use depends on which engine you enable and which weights you download.
