#pragma once
// pipeline-synth.h: ACE-Step synthesis pipeline
//
// Loads DiT + TextEncoder + CondEncoder + VAE once, then generates audio
// from enriched requests (output of pipeline-lm or pre-filled JSON).

#include "request.h"

#include <cstdlib>

struct AceSynth;

struct AceSynthParams {
    const char * text_encoder_path;  // Qwen3 text encoder GGUF (required)
    const char * dit_path;           // DiT GGUF (required)
    const char * vae_path;           // VAE GGUF (NULL = no audio decode, latent only)
    const char * lora_path;          // LoRA adapter path (NULL = no lora)
    float        lora_scale;         // 1.0
    bool         use_fa;             // flash attention (default: true)
    bool         clamp_fp16;         // clamp hidden states to FP16 range (default: false)
    int          vae_chunk;          // latent frames per tile (default: 256)
    int          vae_overlap;        // overlap frames per side (default: 64)
    const char * dump_dir;           // intermediate tensor dump dir (NULL = disabled)
};

// Output audio buffer. Caller must free with ace_audio_free().
struct AceAudio {
    float * samples;      // planar stereo [L0..LN, R0..RN]
    int     n_samples;    // per channel
    int     sample_rate;  // always 48000
};

void ace_synth_default_params(AceSynthParams * p);

// Load all models. NULL on failure.
AceSynth * ace_synth_load(const AceSynthParams * params);

// Generate audio from N requests in a single GPU batch.
// reqs[batch_n]: each request has its own caption, lyrics, metadata, audio_codes, and seed.
//   Text encoding runs per-element, results are padded and stacked for one DiT batch pass.
//   seed must be resolved (non-negative) before calling this function.
//   The first request (reqs[0]) is used for shared params (mode, duration, DiT settings).
// src_audio: interleaved stereo 48kHz source content, NULL for text2music.
// src_len: samples per channel for src_audio.
// ref_audio: interleaved stereo 48kHz timbre reference, NULL = silence (no timbre conditioning).
// ref_len: samples per channel for ref_audio.
// batch_n: number of requests (1..9).
// out[batch_n] allocated by caller, filled with audio buffers.
// cancel/cancel_data: abort callback, polled between DiT steps. NULL = never cancel.
// Returns 0 on success, -1 on error or cancellation.
int ace_synth_generate(AceSynth *         ctx,
                       const AceRequest * reqs,
                       const float *      src_audio,
                       int                src_len,
                       const float *      ref_audio,
                       int                ref_len,
                       int                batch_n,
                       AceAudio *         out,
                       bool (*cancel)(void *) = nullptr,
                       void * cancel_data     = nullptr);

void ace_audio_free(AceAudio * audio);
void ace_synth_free(AceSynth * ctx);
