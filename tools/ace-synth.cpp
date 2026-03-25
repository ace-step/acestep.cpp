// ace-synth.cpp: ACE-Step synthesis CLI
// Thin wrapper: parses args, calls pipeline-synth, writes output files.

#include "audio-io.h"
#include "pipeline-synth.h"
#include "request.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s --request <json...> --embedding <gguf> --dit <gguf> --vae <gguf> [options]\n\n"
            "Required:\n"
            "  --request <json...>     One or more request JSONs (from ace-lm --request)\n"
            "  --embedding <gguf>      Embedding GGUF file\n"
            "  --dit <gguf>            DiT GGUF file\n"
            "  --vae <gguf>            VAE GGUF file\n\n"
            "Reference audio:\n"
            "  --src-audio <file>      Source audio (WAV or MP3, any sample rate)\n\n"
            "LoRA:\n"
            "  --lora <path>           LoRA safetensors file or directory\n"
            "  --lora-scale <float>    LoRA scaling factor (default: 1.0)\n\n"
            "Output:\n"
            "  Default: MP3 at 128 kbps. input.json -> input0.mp3, input1.mp3, ...\n"
            "  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)\n"
            "  --wav                   Output WAV instead of MP3\n\n"
            "VAE tiling (memory control):\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 256)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors\n",
            prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::vector<const char *> request_paths;
    const char *              text_enc_gguf  = NULL;
    const char *              dit_gguf       = NULL;
    const char *              vae_gguf       = NULL;
    const char *              src_audio_path = NULL;
    const char *              dump_dir       = NULL;
    const char *              lora_path      = NULL;
    float                     lora_scale     = 1.0f;
    bool                      use_fa         = true;
    bool                      clamp_fp16     = false;
    int                       vae_chunk      = 256;
    int                       vae_overlap    = 64;
    bool                      output_wav     = false;  // default MP3, --wav forces WAV
    int                       mp3_kbps       = 128;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--request")) {
            // Collect all following non-option args
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                request_paths.push_back(argv[++i]);
            }
        } else if (!strcmp(argv[i], "--embedding") && i + 1 < argc) {
            text_enc_gguf = argv[++i];
        } else if (!strcmp(argv[i], "--dit") && i + 1 < argc) {
            dit_gguf = argv[++i];
        } else if (!strcmp(argv[i], "--vae") && i + 1 < argc) {
            vae_gguf = argv[++i];
        } else if (!strcmp(argv[i], "--src-audio") && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--lora") && i + 1 < argc) {
            lora_path = argv[++i];
        } else if (!strcmp(argv[i], "--lora-scale") && i + 1 < argc) {
            lora_scale = (float) atof(argv[++i]);
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(argv[i], "--no-fa")) {
            use_fa = false;
        } else if (!strcmp(argv[i], "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            vae_overlap = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--wav")) {
            output_wav = true;
        } else if (!strcmp(argv[i], "--mp3-bitrate") && i + 1 < argc) {
            mp3_kbps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (request_paths.empty()) {
        fprintf(stderr, "[CLI] ERROR: --request required\n");
        usage(argv[0]);
        return 1;
    }
    if (!dit_gguf) {
        fprintf(stderr, "[CLI] ERROR: --dit required\n");
        usage(argv[0]);
        return 1;
    }
    if (!text_enc_gguf) {
        fprintf(stderr, "[CLI] ERROR: --embedding required\n");
        usage(argv[0]);
        return 1;
    }

    // Load models
    AceSynthParams params;
    ace_synth_default_params(&params);
    params.text_encoder_path = text_enc_gguf;
    params.dit_path          = dit_gguf;
    params.vae_path          = vae_gguf;
    params.lora_path         = lora_path;
    params.lora_scale        = lora_scale;
    params.use_fa            = use_fa;
    params.clamp_fp16        = clamp_fp16;
    params.vae_chunk         = vae_chunk;
    params.vae_overlap       = vae_overlap;
    params.dump_dir          = dump_dir;

    AceSynth * ctx = ace_synth_load(&params);
    if (!ctx) {
        return 1;
    }

    // Read source audio (cover/lego mode)
    float * src_interleaved = NULL;
    int     src_len         = 0;
    if (src_audio_path) {
        if (!vae_gguf) {
            fprintf(stderr, "[Cover] ERROR: --src-audio requires --vae\n");
            ace_synth_free(ctx);
            return 1;
        }
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Cover] FATAL: cannot read --src-audio %s\n", src_audio_path);
            ace_synth_free(ctx);
            return 1;
        }
        fprintf(stderr, "[Cover] Source audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        // VAE expects interleaved [L0,R0,L1,R1,...], convert from planar
        src_interleaved = (float *) malloc((size_t) T_audio * 2 * sizeof(float));
        for (int t = 0; t < T_audio; t++) {
            src_interleaved[t * 2 + 0] = planar[t];
            src_interleaved[t * 2 + 1] = planar[T_audio + t];
        }
        free(planar);
        src_len = T_audio;
    }

    // Parse all requests
    int                      batch_n = (int) request_paths.size();
    std::vector<AceRequest>  reqs(batch_n);
    std::vector<std::string> basenames(batch_n);
    for (int ri = 0; ri < batch_n; ri++) {
        const char * rpath = request_paths[ri];
        request_init(&reqs[ri]);
        if (!request_parse(&reqs[ri], rpath)) {
            fprintf(stderr, "[Request] FATAL: failed to parse %s\n", rpath);
            ace_synth_free(ctx);
            free(src_interleaved);
            return 1;
        }
        request_dump(&reqs[ri], stderr);
        if (reqs[ri].caption.empty() && reqs[ri].lego.empty()) {
            fprintf(stderr, "[Request] FATAL: caption is empty in %s\n", rpath);
            ace_synth_free(ctx);
            free(src_interleaved);
            return 1;
        }
        // output basename: strip .json suffix
        basenames[ri] = rpath;
        size_t dot    = basenames[ri].rfind(".json");
        if (dot != std::string::npos) {
            basenames[ri] = basenames[ri].substr(0, dot);
        }
    }
    fprintf(stderr, "[Pipeline] Batch: %d request(s)\n", batch_n);

    // expand synth_batch_size: duplicate each request for N DiT variations.
    // resolve seeds and assign consecutive seeds per copy.
    // track local index per original request for output naming.
    std::vector<int> synth_indices;
    {
        std::vector<AceRequest>  exp_reqs;
        std::vector<std::string> exp_names;
        for (int ri = 0; ri < batch_n; ri++) {
            int sbs = reqs[ri].synth_batch_size;
            if (sbs < 1) {
                sbs = 1;
            }
            // resolve seed once per original request
            long long base_seed = reqs[ri].seed;
            if (base_seed < 0) {
                std::random_device rd;
                base_seed = (long long) rd();
            }
            for (int i = 0; i < sbs; i++) {
                AceRequest copy = reqs[ri];
                copy.seed       = base_seed + i;
                exp_reqs.push_back(copy);
                exp_names.push_back(basenames[ri]);
                synth_indices.push_back(i);
            }
        }
        reqs      = std::move(exp_reqs);
        basenames = std::move(exp_names);
        batch_n   = (int) reqs.size();
        if (batch_n > 1) {
            fprintf(stderr, "[Pipeline] Expanded to %d (synth_batch_size)\n", batch_n);
        }
    }

    // Generate all in one GPU batch
    std::vector<AceAudio> audio(batch_n);
    if (ace_synth_generate(ctx, reqs.data(), src_interleaved, src_len, batch_n, audio.data()) != 0) {
        fprintf(stderr, "[Pipeline] ERROR: generation failed\n");
        free(src_interleaved);
        ace_synth_free(ctx);
        return 1;
    }

    // Write output files
    for (int b = 0; b < batch_n; b++) {
        if (!audio[b].samples) {
            continue;
        }
        const char * ext = output_wav ? ".wav" : ".mp3";
        char         out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s%d%s", basenames[b].c_str(), synth_indices[b], ext);
        if (!audio_write(out_path, audio[b].samples, audio[b].n_samples, 48000, mp3_kbps)) {
            fprintf(stderr, "[Batch%d] FATAL: failed to write %s\n", b, out_path);
        }
        ace_audio_free(&audio[b]);
    }

    free(src_interleaved);
    ace_synth_free(ctx);
    fprintf(stderr, "[Pipeline] All done\n");
    return 0;
}
