// coderai embedding server for dinov2.cpp
//
// Loads a DINOv2 GGUF once, then serves a stdin/stdout loop: each input line
// is an image path; each output line is JSON {"embedding":[...]} (the CLS
// token after the final layernorm — identical to HF Dinov2Model's
// pooler-input, so vectors match the transformers 'vision' backend) or
// {"error":"..."}. Preprocessing mirrors the HF AutoImageProcessor for
// facebook/dinov2-*: shortest side -> 256 (bicubic), center-crop 224,
// ImageNet mean/std normalization.
//
// GPU: built with GGML_VULKAN this runs the graph on the Vulkan device
// (radeon); tensors are read back with ggml_backend_tensor_get (device-safe,
// unlike the upstream tools' ggml_get_data_f32). DINOV2_FORCE_CPU=1 forces
// the CPU backend.

#include "dinov2.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

static cv::Mat hf_preprocess(const cv::Mat &bgr) {
    // shortest side -> 256, bicubic (matches HF resize)
    const int short_side = std::min(bgr.cols, bgr.rows);
    const double scale = 256.0 / short_side;
    cv::Mat resized;
    cv::resize(bgr, resized,
               cv::Size(int(round(bgr.cols * scale)), int(round(bgr.rows * scale))),
               0, 0, cv::INTER_CUBIC);
    // center crop 224x224
    const int x = (resized.cols - 224) / 2;
    const int y = (resized.rows - 224) / 2;
    cv::Mat crop = resized(cv::Rect(x, y, 224, 224)).clone();
    // float [0,1] + ImageNet standardization (channels are BGR here; the
    // mean/std constants are indexed reversed exactly like dino_preprocess)
    cv::Mat image;
    crop.convertTo(image, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - IMAGENET_DEFAULT_MEAN[2 - i])
                      / IMAGENET_DEFAULT_STD[2 - i];
    }
    cv::merge(channels, image);
    return image;
}

static bool predict_cls(dino_model &model, const cv::Mat &img,
                        const dino_params &params, ggml_gallocr_t allocr,
                        std::vector<float> &out) {
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                          ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context *ctx_cgraph = ggml_init(params0);
    struct ggml_cgraph *gf = build_graph(img.size(), ctx_cgraph, model, params);
    ggml_gallocr_alloc_graph(allocr, gf);

    // input image, planar RGB
    struct ggml_tensor *input = ggml_graph_get_tensor(gf, "input");
    std::vector<float> planar(img.total() * 3);
    float *dst = planar.data();
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(img, bgr_channels);
    std::vector<cv::Mat> rgb_planar = {
        cv::Mat(img.rows, img.cols, CV_32F, dst + 0 * img.total()),
        cv::Mat(img.rows, img.cols, CV_32F, dst + 1 * img.total()),
        cv::Mat(img.rows, img.cols, CV_32F, dst + 2 * img.total()),
    };
    bgr_channels[2].copyTo(rgb_planar[0]);
    bgr_channels[1].copyTo(rgb_planar[1]);
    bgr_channels[0].copyTo(rgb_planar[2]);
    ggml_backend_tensor_set(input, planar.data(), 0, ggml_nbytes(input));

    // interpolated position embeddings — copy the source tensor to HOST first
    // (on a GPU backend ->data points at device memory; upstream reads it
    // directly and only works on CPU)
    const struct ggml_tensor *pos_embed =
        ggml_get_tensor(model.ctx, "embeddings.position_embeddings");
    std::vector<float> pos_host(ggml_nelements(pos_embed));
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(pos_embed),
                            pos_host.data(), 0, ggml_nbytes(pos_embed));
    const std::vector<float> pos_fixed =
        interpolate_pos_embed(img.size(), pos_host.data(), model.hparams);
    struct ggml_tensor *pos_embed_fixed =
        ggml_graph_get_tensor(gf, "pos_embed_fixed");
    ggml_backend_tensor_set(pos_embed_fixed, pos_fixed.data(), 0,
                            ggml_nbytes(pos_embed_fixed));

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx_cgraph);
        return false;
    }

    struct ggml_tensor *cls = ggml_graph_get_tensor(gf, "cls_token");
    out.resize(model.hparams.hidden_size);
    ggml_backend_tensor_get(cls, out.data(), 0, out.size() * sizeof(float));
    ggml_free(ctx_cgraph);
    return true;
}

int main(int argc, char **argv) {
    ggml_time_init();
    dino_params params;
    params.classify = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) params.model = argv[++i];
        else if (arg == "-t" && i + 1 < argc) params.n_threads = std::stoi(argv[++i]);
    }

    dino_model model;
    if (!dino_model_load(cv::Size(224, 224), params.model, model, params)) {
        fprintf(stderr, "embed: failed to load model '%s'\n", params.model.c_str());
        return 1;
    }
    ggml_gallocr_t allocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    fprintf(stderr, "embed: ready (hidden=%d)\n", model.hparams.hidden_size);
    printf("{\"ready\":true,\"hidden\":%d}\n", model.hparams.hidden_size);
    fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        cv::Mat raw = cv::imread(line, cv::IMREAD_COLOR);
        if (raw.empty()) {
            printf("{\"error\":\"cannot read image\"}\n");
            fflush(stdout);
            continue;
        }
        cv::Mat img = hf_preprocess(raw);
        std::vector<float> emb;
        if (!predict_cls(model, img, params, allocr, emb)) {
            printf("{\"error\":\"graph compute failed\"}\n");
            fflush(stdout);
            continue;
        }
        std::string out = "{\"embedding\":[";
        char buf[32];
        for (size_t i = 0; i < emb.size(); ++i) {
            snprintf(buf, sizeof(buf), i ? ",%.6g" : "%.6g", emb[i]);
            out += buf;
        }
        out += "]}";
        puts(out.c_str());
        fflush(stdout);
    }

    ggml_gallocr_free(allocr);
    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
