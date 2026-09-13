// Emit golden vectors from the C++ implementation, for the Python port to match.
//
// WHY THIS EXISTS
//   The existing C++ test suite is almost entirely DIRECTIONAL: it asserts
//   things like `conflicted_cert.lower_log_bf < clean_cert.lower_log_bf` and
//   `cert.structure_lower_log_lr > 0.0`. Very little of it pins a number.
//
//   That is fine as a regression net for C++ refactors, but it is nearly
//   useless as a migration contract: a Python port could compute completely
//   different values and still pass every one of those assertions. Sign and
//   ordering are a weak specification.
//
//   So before porting anything, freeze the actual numbers. This program walks a
//   grid of inputs through the real C++ entry points and writes JSON. The Python
//   tests then assert equality against that JSON to a tight tolerance, which
//   turns "behaves similarly" into "computes the same function".
//
// WHAT IT CAN COVER
//   Only the pure-logic layer -- which is exactly the layer being migrated
//   first, and conveniently the layer that does not need htslib or abPOA. These
//   sources link standalone:
//       src/pipeline/mechanistic_evidence.cpp
//       src/pipeline/decision_policy.cpp
//       src/pipeline/event_explanation.cpp
//       src/component/te_sequence_explainer.cpp
//
// BUILD (no CMake, no htslib required)
//   g++ -std=c++17 -O1 -I../../include \
//       dump_oracle.cpp \
//       ../../src/pipeline/mechanistic_evidence.cpp \
//       ../../src/pipeline/decision_policy.cpp \
//       ../../src/pipeline/event_explanation.cpp \
//       ../../src/component/te_sequence_explainer.cpp \
//       -o dump_oracle
//   ./dump_oracle > ../tests/oracle/cpp_reference.json
//
// REGENERATE IT whenever the C++ decision layer changes on purpose. A diff in
// this JSON is the signal that the contract moved, and it should never move by
// accident.

#include "conformal_selector.h"
#include "decision_policy.h"
#include "event_explanation.h"
#include "mechanistic_evidence.h"
#include "null_control.h"
#include "pipeline.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_indent = 0;

void ind() {
    for (int i = 0; i < g_indent; ++i) {
        std::printf("  ");
    }
}

// Full round-trip precision: the point is exact agreement, so do not round.
void kv(const char* key, double value, bool comma = true) {
    ind();
    std::printf("\"%s\": %.17g%s\n", key, value, comma ? "," : "");
}

void kv(const char* key, int value, bool comma = true) {
    ind();
    std::printf("\"%s\": %d%s\n", key, value, comma ? "," : "");
}

void kv(const char* key, const std::string& value, bool comma = true) {
    ind();
    std::printf("\"%s\": \"%s\"%s\n", key, value.c_str(), comma ? "," : "");
}

// ---------------------------------------------------------------- fixtures
placer::EventExistenceEvidence existence(
    int32_t alt, int32_t split, int32_t indel, int32_t lclip, int32_t rclip,
    int32_t ref, double af, int32_t gq) {
    placer::EventExistenceEvidence e;
    e.alt_struct_reads = alt;
    e.alt_split_reads = split;
    e.alt_indel_reads = indel;
    e.alt_left_clip_reads = lclip;
    e.alt_right_clip_reads = rclip;
    e.ref_span_reads = ref;
    e.af = af;
    e.gq = gq;
    e.score = 2.0;
    return e;
}

placer::EventSegmentationEvidence segmentation(
    bool pair_valid, bool left, bool right, int32_t insert_len) {
    placer::EventSegmentationEvidence s;
    s.has_consensus = true;
    s.has_insert_seq = true;
    s.has_left_flank = left;
    s.has_right_flank = right;
    s.pair_valid = pair_valid;
    s.insert_len = insert_len;
    s.score = 1.5;
    s.qc = "PASS_EVENT_SEGMENTATION";
    return s;
}

placer::BoundaryEvidence boundary(
    bool geometry, bool canonical, bool consistent, const char* type,
    int32_t len) {
    placer::BoundaryEvidence b;
    b.geometry_defined = geometry;
    b.canonical_pass = canonical;
    b.evidence_consistent = consistent;
    b.boundary_type = type;
    b.boundary_len = len;
    b.score = 1.0;
    b.qc = "PASS_BOUNDARY_TSD";
    return b;
}

placer::TEAlignmentEvidence te_alignment(
    double identity, double coverage, double margin, const char* qc,
    const char* model, double model_score, const char* confidence,
    double residual_fraction) {
    placer::TEAlignmentEvidence t;
    t.pass = true;
    t.best_family = "L1";
    t.best_subfamily = "L1HS";
    t.best_identity = identity;
    t.best_query_coverage = coverage;
    t.cross_family_margin = margin;
    t.qc_reason = qc;
    t.sequence_model_label = model;
    t.sequence_model_score = model_score;
    t.annotation_confidence = confidence;
    t.annotation_residual_fraction = residual_fraction;
    return t;
}

struct Scenario {
    const char* name;
    placer::EventExistenceEvidence ex;
    placer::EventSegmentationEvidence seg;
    placer::TEAlignmentEvidence te;
    placer::BoundaryEvidence bd;
};

std::vector<Scenario> scenarios() {
    std::vector<Scenario> out;
    // The canonical strong case from tests/test_mechanistic_evidence.cpp.
    out.push_back({"strong_resolved_te",
        existence(18, 8, 3, 5, 5, 0, 0.65, 60),
        segmentation(true, true, true, 320),
        te_alignment(0.96, 0.90, 0.18, "PASS_INSERT_TE_ALIGNMENT",
                     "TE_MODEL_IN_DISTRIBUTION", 0.35, "HIGH", 0.0),
        boundary(true, true, true, "TSD", 12)});
    // Same, but heavily opposed by reference-spanning reads.
    out.push_back({"strong_with_reference_conflict",
        existence(18, 8, 3, 5, 5, 18, 0.65, 60),
        segmentation(true, true, true, 320),
        te_alignment(0.96, 0.90, 0.18, "PASS_INSERT_TE_ALIGNMENT",
                     "TE_MODEL_IN_DISTRIBUTION", 0.35, "HIGH", 0.0),
        boundary(true, true, true, "TSD", 12)});
    // Sequence-composition outlier: the insert is not in the TE distribution.
    out.push_back({"sequence_model_outlier",
        existence(18, 8, 3, 5, 5, 0, 0.65, 60),
        segmentation(true, true, true, 320),
        te_alignment(0.96, 0.90, 0.18, "PASS_INSERT_TE_ALIGNMENT",
                     "TE_MODEL_OUTLIER", -0.80, "HIGH", 0.0),
        boundary(true, true, true, "TSD", 12)});
    // Family resolved but not subfamily.
    out.push_back({"family_only",
        existence(12, 5, 2, 3, 3, 1, 0.45, 42),
        segmentation(true, true, true, 280),
        te_alignment(0.88, 0.72, 0.03, "PASS_INSERT_TE_ALIGNMENT_FAMILY_ONLY",
                     "TE_MODEL_EDGE", 0.05, "MEDIUM", 0.10),
        boundary(true, false, true, "BLUNT", 0)});
    // One-sided: insert sequence present, only one flank anchored.
    out.push_back({"one_sided_rescue",
        existence(6, 2, 1, 2, 0, 2, 0.30, 28),
        segmentation(false, true, false, 410),
        te_alignment(0.93, 0.80, 0.09, "PASS_INSERT_TE_ALIGNMENT",
                     "TE_MODEL_IN_DISTRIBUTION", 0.20, "HIGH", 0.05),
        boundary(true, false, true, "SMALL_DEL", 3)});
    // Low identity: the alignment is TE-like but below the identity gate.
    out.push_back({"low_identity",
        existence(9, 4, 1, 2, 2, 3, 0.35, 33),
        segmentation(true, true, true, 260),
        te_alignment(0.71, 0.55, 0.01, "TE_ALIGNMENT_LOW_IDENTITY",
                     "TE_MODEL_EDGE", -0.10, "LOW", 0.30),
        boundary(true, false, false, "NONE", 0)});
    // Unknown TE: TE-like sequence with no confident family.
    out.push_back({"unknown_te",
        existence(14, 6, 2, 4, 4, 1, 0.55, 51),
        segmentation(true, true, true, 350),
        te_alignment(0.90, 0.84, 0.00, "PASS_INSERT_TE_ALIGNMENT_UNKNOWN",
                     "TE_MODEL_IN_DISTRIBUTION", 0.28, "MEDIUM", 0.18),
        boundary(true, true, true, "TSD", 15)});
    // Artifact-shaped: no geometry, high residual, opposed.
    out.push_back({"artifact_shaped",
        existence(4, 1, 0, 1, 0, 11, 0.15, 12),
        segmentation(false, false, false, 120),
        te_alignment(0.78, 0.40, 0.00, "TE_ALIGNMENT_LOW_IDENTITY",
                     "TE_MODEL_OUTLIER", -0.90, "LOW", 0.55),
        boundary(false, false, false, "NONE", 0)});
    return out;
}

void dump_certificates() {
    ind();
    std::printf("\"certificates\": [\n");
    ++g_indent;
    const auto cases = scenarios();
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& s = cases[i];
        const auto cert = placer::build_mechanistic_evidence_certificate(
            s.ex, s.seg, s.te, s.bd, nullptr);
        ind();
        std::printf("{\n");
        ++g_indent;
        kv("name", std::string(s.name));
        kv("event_lower_log_lr", cert.event_lower_log_lr);
        kv("independent_lower_log_lr", cert.independent_lower_log_lr);
        kv("sequence_lower_log_lr", cert.sequence_lower_log_lr);
        kv("structure_lower_log_lr", cert.structure_lower_log_lr);
        kv("boundary_lower_log_lr", cert.boundary_lower_log_lr);
        kv("ref_conflict_lower_log_lr", cert.ref_conflict_lower_log_lr);
        kv("structure_te_log_evidence", cert.structure_te_log_evidence);
        kv("structure_nonte_log_evidence", cert.structure_nonte_log_evidence);
        kv("structure_artifact_log_evidence", cert.structure_artifact_log_evidence);
        kv("raw_log_bf_te_vs_artifact", cert.raw_log_bf_te_vs_artifact);
        kv("raw_log_bf_te_vs_non_te", cert.raw_log_bf_te_vs_non_te);
        kv("lower_log_bf_te_vs_artifact", cert.lower_log_bf_te_vs_artifact);
        kv("lower_log_bf_te_vs_non_te", cert.lower_log_bf_te_vs_non_te);
        kv("mechanistic_support_signal", cert.mechanistic_support_signal);
        kv("ref_conflict_signal", cert.ref_conflict_signal);
        kv("artifact_context_signal", cert.artifact_context_signal);
        kv("ambiguity_width", cert.ambiguity_width);
        kv("n_blocks", static_cast<int>(cert.blocks.size()));
        ind();
        std::printf("\"blocks\": [\n");
        ++g_indent;
        for (size_t b = 0; b < cert.blocks.size(); ++b) {
            const auto& blk = cert.blocks[b];
            ind();
            std::printf("{\"name\": \"%s\", \"raw_signal\": %.17g, "
                        "\"te_vs_artifact\": %.17g, \"te_vs_non_te\": %.17g, "
                        "\"ambiguity_width\": %.17g}%s\n",
                        blk.name.c_str(), blk.raw_signal,
                        blk.lower_log_lr_te_vs_artifact,
                        blk.lower_log_lr_te_vs_non_te, blk.ambiguity_width,
                        b + 1 < cert.blocks.size() ? "," : "");
        }
        --g_indent;
        ind();
        std::printf("],\n");
        const placer::PriorInterval prior;
        const auto lfdr =
            placer::evaluate_robust_mechanistic_lfdr(cert, prior, 0.10);
        kv("robust_worst_case_lfdr", lfdr.worst_case_lfdr);
        kv("robust_qc", lfdr.qc, false);
        --g_indent;
        ind();
        std::printf("}%s\n", i + 1 < cases.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");
}

void dump_genotypes() {
    ind();
    std::printf("\"genotypes\": [\n");
    ++g_indent;
    struct GtCase { int32_t alt, ref; double err, rho; };
    const std::vector<GtCase> cases = {
        {0, 10, 0.02, 0.02}, {1, 9, 0.02, 0.02}, {3, 3, 0.02, 0.02},
        {3, 3, 0.10, 0.02}, {8, 8, 0.02, 0.02}, {8, 8, 0.10, 0.02},
        {10, 0, 0.02, 0.02}, {2, 0, 0.02, 0.02}, {5, 1, 0.02, 0.02},
        {20, 20, 0.02, 0.02}, {6, 30, 0.02, 0.02}, {3, 3, 0.02, 0.20},
        {8, 8, 0.02, 0.20}, {40, 4, 0.02, 0.02},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        placer::EventGenotypeInput in;
        in.alt_struct_reads = cases[i].alt;
        in.ref_span_reads = cases[i].ref;
        in.error_rate = cases[i].err;
        in.overdispersion = cases[i].rho;
        in.event_length = 320;
        const auto d = placer::genotype_event_from_alt_vs_ref(in);
        ind();
        std::printf("{\"alt\": %d, \"ref\": %d, \"error_rate\": %.17g, "
                    "\"overdispersion\": %.17g, \"best_gt\": \"%s\", "
                    "\"allele_fraction\": %.17g, \"gq\": %d, \"depth\": %d, "
                    "\"best_nonref_minus_ref_ll\": %.17g, \"pass\": %s}%s\n",
                    cases[i].alt, cases[i].ref, cases[i].err, cases[i].rho,
                    d.best_gt.c_str(), d.allele_fraction, d.gq, d.depth,
                    d.best_nonref_minus_ref_ll, d.pass ? "true" : "false",
                    i + 1 < cases.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");
}

void dump_overdispersion() {
    ind();
    std::printf("\"overdispersion\": [\n");
    ++g_indent;
    // Deterministic synthetic count sets, so the Python port can match exactly.
    std::vector<std::vector<placer::AltDepthObservation>> sets;
    sets.push_back({});                                    // empty -> fallback
    sets.push_back({{5, 10}});                             // one site
    sets.push_back({{5, 10}, {6, 12}, {4, 9}, {7, 15}});
    {
        std::vector<placer::AltDepthObservation> big;
        for (int i = 0; i < 200; ++i) {
            big.push_back({(i % 7) + 3, (i % 11) + 10});
        }
        sets.push_back(big);
    }
    for (size_t i = 0; i < sets.size(); ++i) {
        const double rho =
            placer::estimate_alt_depth_overdispersion(sets[i], 0.02);
        ind();
        std::printf("{\"n_sites\": %d, \"rho\": %.17g}%s\n",
                    static_cast<int>(sets[i].size()), rho,
                    i + 1 < sets.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");
}

void dump_dependency_penalty() {
    ind();
    std::printf("\"dependency_penalty\": [\n");
    ++g_indent;
    // Fixed, reproducible inputs -- no RNG, so Python can feed the same list.
    struct DpCase {
        const char* name;
        std::vector<double> art;
        std::vector<double> non;
        double q;
        size_t m;
    };
    std::vector<DpCase> cases;
    {
        std::vector<double> bulk;
        for (int i = 0; i < 500; ++i) {
            bulk.push_back(-6.0 + 0.01 * i);
        }
        cases.push_back({"ramp_500", bulk, bulk, 0.10, 1000});
        std::vector<double> with_tail = bulk;
        for (int i = 0; i < 50; ++i) {
            with_tail.push_back(4.0 + 0.02 * i);
        }
        cases.push_back({"ramp_with_right_tail", with_tail, with_tail, 0.10, 1000});
        cases.push_back({"tiny_two", {0.1, 0.3}, {0.1, 0.3}, 0.10, 1000});
        cases.push_back({"single_row", {0.5}, {0.5}, 0.10, 1000});
        cases.push_back({"empty", {}, {}, 0.10, 1000});
        cases.push_back({"asymmetric_sides", with_tail, bulk, 0.05, 250});
    }
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const auto est = placer::estimate_dependency_penalty(
            c.art, c.non, c.q, c.m);
        ind();
        std::printf("{\"name\": \"%s\", \"q\": %.17g, \"m\": %d, "
                    "\"n_art\": %d, \"n_non\": %d, \"cap_log\": %.17g, "
                    "\"sigma_mean_art\": %.17g, \"sigma_upper_art\": %.17g, "
                    "\"log_penalty_art\": %.17g, \"sigma_mean_non\": %.17g, "
                    "\"sigma_upper_non\": %.17g, \"log_penalty_non\": %.17g, "
                    "\"null_count\": %d, \"estimated\": %s}%s\n",
                    c.name, c.q, static_cast<int>(c.m),
                    static_cast<int>(c.art.size()),
                    static_cast<int>(c.non.size()), est.cap_log,
                    est.vs_artifact.sigma_mean, est.vs_artifact.sigma_upper,
                    est.vs_artifact.log_penalty, est.vs_non_te.sigma_mean,
                    est.vs_non_te.sigma_upper, est.vs_non_te.log_penalty,
                    est.null_count, est.estimated ? "true" : "false",
                    i + 1 < cases.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");
}

void dump_sequence_structure() {
    ind();
    std::printf("\"sequence_structure\": [\n");
    ++g_indent;
    struct SsCase {
        const char* name;
        std::string seq;
        const char* qc;
        double identity, coverage, residual, masked, margin;
        const char* model;
        double model_score;
    };
    const std::vector<SsCase> cases = {
        {"core_plus_polya", std::string(220, 'G') + std::string(30, 'A'),
         "PASS_INSERT_TE_ALIGNMENT", 0.96, 0.88, 0.12, 0.0, 0.18,
         "TE_MODEL_IN_DISTRIBUTION", 0.35},
        {"core_only", std::string(250, 'G'),
         "PASS_INSERT_TE_ALIGNMENT", 0.96, 0.99, 0.01, 0.0, 0.18,
         "TE_MODEL_IN_DISTRIBUTION", 0.35},
        {"core_transduction_polya",
         std::string(200, 'G') + "CTAGGCATTCGAATCGGATCCTAGGCATTCGAAT" +
             std::string(25, 'A'),
         "PASS_INSERT_TE_ALIGNMENT", 0.95, 0.72, 0.28, 0.02, 0.15,
         "TE_MODEL_IN_DISTRIBUTION", 0.30},
        {"low_coverage_high_residual", std::string(140, 'G') + std::string(110, 'C'),
         "PASS_INSERT_TE_ALIGNMENT", 0.92, 0.55, 0.45, 0.05, 0.06,
         "TE_MODEL_EDGE", 0.0},
        {"masked_residual", std::string(150, 'G') + std::string(100, 'T'),
         "PASS_INSERT_TE_ALIGNMENT", 0.90, 0.60, 0.40, 0.35, 0.04,
         "TE_MODEL_EDGE", -0.05},
        {"unknown_family", std::string(200, 'G') + std::string(20, 'A'),
         "PASS_INSERT_TE_ALIGNMENT_UNKNOWN", 0.90, 0.84, 0.18, 0.0, 0.0,
         "TE_MODEL_IN_DISTRIBUTION", 0.28},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const auto ex = placer::explain_te_sequence_structure(
            c.seq, c.qc, "L1", "L1HS", c.identity, c.coverage, c.residual,
            c.masked, c.margin, 0.0, c.model, c.model_score);
        ind();
        std::printf("{\"name\": \"%s\", \"insert_len\": %d, "
                    "\"te_structure_log_evidence\": %.17g, "
                    "\"nonte_structure_log_evidence\": %.17g, "
                    "\"artifact_structure_log_evidence\": %.17g, "
                    "\"structure_path_confidence\": %.17g, "
                    "\"polyA_posterior\": %.17g, "
                    "\"transduction_posterior\": %.17g, "
                    "\"te_core_coverage\": %.17g, "
                    "\"unexplained_high_complexity_bp\": %d}%s\n",
                    c.name, static_cast<int>(c.seq.size()),
                    ex.te_structure_log_evidence,
                    ex.nonte_structure_log_evidence,
                    ex.artifact_structure_log_evidence,
                    ex.structure_path_confidence, ex.polyA_posterior,
                    ex.transduction_posterior, ex.te_core_coverage,
                    ex.unexplained_high_complexity_bp,
                    i + 1 < cases.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("]\n");
}

// ===================================================== event explanation
placer::ExplanationResidual residual(int s, int m, int u, int b, int r,
                                     int rc, int a, int la, int pc) {
    placer::ExplanationResidual out;
    out.structural_conflicts = s;
    out.missing_required_components = m;
    out.unexplained_high_complexity_bases = u;
    out.breakpoint_disagreement_bp = b;
    out.read_assignment_conflicts = r;
    out.reference_counterevidence = rc;
    out.artifact_evidence = a;
    out.label_ambiguity = la;
    out.path_complexity = pc;
    return out;
}

placer::EventExplanation explanation(placer::ExplanationKind kind,
                                     const placer::ExplanationResidual& res,
                                     const char* family, const char* subfamily) {
    placer::EventExplanation out;
    out.kind = kind;
    out.residual = res;
    out.family = family;
    out.subfamily = subfamily;
    return out;
}

void dump_event_explanations() {
    ind();
    std::printf("\"event_explanations\": [\n");
    ++g_indent;

    struct Case {
        const char* name;
        std::vector<placer::EventExplanation> set;
        bool closed;
    };
    std::vector<Case> cases;

    // TE uniquely dominates, closed breakpoints.
    cases.push_back({"te_dominates_closed", {
        explanation(placer::ExplanationKind::kTe, residual(0,0,0,0,0,0,0,0,1), "L1", "L1HS"),
        explanation(placer::ExplanationKind::kInsertionNonTe, residual(0,0,160,0,0,0,0,0,1), "NA", "NA"),
        explanation(placer::ExplanationKind::kReference, residual(1,1,0,0,3,3,0,0,0), "NA", "NA"),
        explanation(placer::ExplanationKind::kArtifact, residual(0,0,106,0,1,3,3,0,1), "NA", "NA"),
    }, true});
    // Same, imprecise.
    cases.push_back({"te_dominates_imprecise", {
        explanation(placer::ExplanationKind::kTe, residual(0,0,0,25,0,0,0,0,1), "L1", "L1HS"),
        explanation(placer::ExplanationKind::kInsertionNonTe, residual(0,0,160,0,0,0,0,0,1), "NA", "NA"),
        explanation(placer::ExplanationKind::kReference, residual(1,1,0,0,3,3,0,0,0), "NA", "NA"),
    }, false});
    // MISNAMED ON PURPOSE, kept because it caught a false assumption: TE has a
    // HIGHER primary residual sum here (2 vs 0), so non-TE dominates and the
    // answer is PASS_NONTE_INSERTION, not TE_AMBIGUOUS.
    cases.push_back({"te_loses_to_nonte", {
        explanation(placer::ExplanationKind::kTe, residual(0,0,0,0,2,0,0,0,1), "L1", "L1HS"),
        explanation(placer::ExplanationKind::kInsertionNonTe, residual(0,0,0,0,0,0,0,0,1), "NA", "NA"),
    }, true});
    // A REAL TE_AMBIGUOUS: identical primary residuals, so neither dominates.
    // TE sorts first only because its kind ordinal is higher than non-TE's,
    // and then fails the unique-dominance test against an equal rival.
    cases.push_back({"te_ambiguous", {
        explanation(placer::ExplanationKind::kTe, residual(0,0,0,0,1,0,0,0,0), "L1", "L1HS"),
        explanation(placer::ExplanationKind::kInsertionNonTe, residual(0,0,0,0,1,0,0,0,0), "NA", "NA"),
    }, true});
    // Missing components -> NO_CALL_INCOMPLETE.
    cases.push_back({"incomplete", {
        explanation(placer::ExplanationKind::kTe, residual(0,1,0,0,0,0,0,0,1), "L1", "L1HS"),
        explanation(placer::ExplanationKind::kReference, residual(1,1,0,0,3,3,0,0,0), "NA", "NA"),
    }, true});
    // Non-TE insertion wins uniquely.
    cases.push_back({"nonte_insertion", {
        explanation(placer::ExplanationKind::kInsertionNonTe, residual(0,0,0,0,0,0,0,0,1), "NA", "NA"),
        explanation(placer::ExplanationKind::kTe, residual(0,0,300,0,0,0,1,1,2), "UNKNOWN", "UNKNOWN"),
        explanation(placer::ExplanationKind::kArtifact, residual(0,0,106,0,1,3,3,0,1), "NA", "NA"),
    }, true});
    // Artifact wins -> REFERENCE_OR_ARTIFACT.
    cases.push_back({"artifact_wins", {
        explanation(placer::ExplanationKind::kArtifact, residual(0,0,0,0,0,0,0,0,1), "NA", "NA"),
        explanation(placer::ExplanationKind::kTe, residual(0,0,400,0,5,9,2,1,2), "UNKNOWN", "UNKNOWN"),
    }, true});
    // TE resolved but family unknown -> emit_unknown_te even when closed.
    cases.push_back({"te_unknown_family_closed", {
        explanation(placer::ExplanationKind::kTe, residual(0,0,0,0,0,0,0,1,2), "UNKNOWN", "UNKNOWN"),
        explanation(placer::ExplanationKind::kReference, residual(1,1,0,0,3,3,0,0,0), "NA", "NA"),
    }, true});

    for (size_t i = 0; i < cases.size(); ++i) {
        const auto d = placer::compare_event_explanations(cases[i].set,
                                                          cases[i].closed);
        ind();
        std::printf("{\"name\": \"%s\", \"closed\": %s, \"final_qc\": \"%s\", "
                    "\"best_kind\": \"%s\", \"emit_te_call\": %s, "
                    "\"emit_unknown_te\": %s, \"emit_evidence_te_call\": %s, "
                    "\"n_alternatives\": %d, \"best_residual\": \"%s\"}%s\n",
                    cases[i].name, cases[i].closed ? "true" : "false",
                    d.final_qc.c_str(),
                    placer::explanation_kind_name(d.best.kind),
                    d.emit_te_call ? "true" : "false",
                    d.emit_unknown_te ? "true" : "false",
                    d.emit_evidence_te_call ? "true" : "false",
                    static_cast<int>(d.alternatives.size()),
                    placer::serialize_residual(d.best.residual).c_str(),
                    i + 1 < cases.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");

    // Pairwise dominance truth table.
    ind();
    std::printf("\"dominance_pairs\": [\n");
    ++g_indent;
    struct Pair { const char* name; placer::ExplanationResidual a, b; };
    const std::vector<Pair> pairs = {
        {"strictly_better_everywhere", residual(0,0,0,0,0,0,0,0,0),
                                        residual(1,1,1,1,1,1,1,1,1)},
        {"equal", residual(1,1,1,0,1,1,1,0,0), residual(1,1,1,0,1,1,1,0,0)},
        {"one_better_one_worse", residual(0,1,0,0,0,0,0,0,0),
                                  residual(1,0,0,0,0,0,0,0,0)},
        {"one_strictly_better_rest_equal", residual(0,1,1,0,1,1,1,0,0),
                                            residual(1,1,1,0,1,1,1,0,0)},
        {"tie_break_fields_ignored", residual(1,1,1,99,1,1,1,99,99),
                                      residual(1,1,1,0,1,1,1,0,0)},
    };
    for (size_t i = 0; i < pairs.size(); ++i) {
        ind();
        std::printf("{\"name\": \"%s\", \"a_dominates_b\": %s, "
                    "\"b_dominates_a\": %s}%s\n", pairs[i].name,
                    placer::dominates_primary_residuals(pairs[i].a, pairs[i].b)
                        ? "true" : "false",
                    placer::dominates_primary_residuals(pairs[i].b, pairs[i].a)
                        ? "true" : "false",
                    i + 1 < pairs.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");
}

// ===================================================== conformal selector
void dump_conformal() {
    ind();
    std::printf("\"conformal\": [\n");
    ++g_indent;

    struct Scenario {
        const char* name;
        int n_nulls;
        int n_candidates;
        int contexts;
        double q;
    };
    const std::vector<Scenario> scenarios = {
        {"pooled_small", 8, 4, 1, 0.10},      // below kMinContextNulls -> pooled
        {"context_split", 40, 6, 2, 0.10},
        {"no_nulls", 0, 3, 1, 0.10},
        {"strict_q", 40, 6, 2, 0.01},
    };

    for (size_t s = 0; s < scenarios.size(); ++s) {
        const auto& sc = scenarios[s];
        placer::ConformalNullSelector selector;
        for (int i = 0; i < sc.n_nulls; ++i) {
            placer::ConformalFeatureVector f;
            f.id = "null_" + std::to_string(i);
            // Deterministic ramp: weak nulls, a few strong ones.
            const double t = static_cast<double>(i) / std::max(1, sc.n_nulls - 1);
            f.pro_te = {2.0 + 6.0 * t, 0.60 + 0.30 * t, 0.50 + 0.40 * t,
                        0.01 + 0.10 * t};
            f.ref_span_reads = 10.0 - 8.0 * t;
            f.context = i % std::max(1, sc.contexts);
            selector.add_null_control(f);
        }
        std::vector<placer::ConformalFeatureVector> candidates;
        for (int i = 0; i < sc.n_candidates; ++i) {
            placer::ConformalFeatureVector c;
            c.id = "cand_" + std::to_string(i);
            const double t = static_cast<double>(i) /
                             std::max(1, sc.n_candidates - 1);
            c.pro_te = {4.0 + 8.0 * t, 0.70 + 0.28 * t, 0.60 + 0.38 * t,
                        0.02 + 0.16 * t};
            c.ref_span_reads = 6.0 - 5.0 * t;
            c.context = i % std::max(1, sc.contexts);
            candidates.push_back(c);
        }
        const auto results = selector.select(candidates, sc.q);
        ind();
        std::printf("{\"name\": \"%s\", \"n_nulls\": %d, \"n_candidates\": %d, "
                    "\"contexts\": %d, \"q\": %.17g, \"results\": [\n",
                    sc.name, sc.n_nulls, sc.n_candidates, sc.contexts, sc.q);
        ++g_indent;
        for (size_t i = 0; i < results.size(); ++i) {
            ind();
            std::printf("{\"id\": \"%s\", \"conformal_p\": %.17g, "
                        "\"by_threshold\": %.17g, \"dominated\": %d, "
                        "\"null_count\": %d, \"pass\": %s, \"qc\": \"%s\"}%s\n",
                        results[i].id.c_str(), results[i].conformal_p,
                        results[i].by_threshold,
                        static_cast<int>(results[i].dominated_null_count),
                        static_cast<int>(results[i].null_count),
                        results[i].pass ? "true" : "false",
                        results[i].qc.c_str(),
                        i + 1 < results.size() ? "," : "");
        }
        --g_indent;
        ind();
        std::printf("]}%s\n", s + 1 < scenarios.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");
}

// ===================================================== null controls
void dump_null_controls() {
    ind();
    std::printf("\"breakpoint_shifts\": [\n");
    ++g_indent;
    struct Case { int bl, br, ws, we, step, maxc; };
    const std::vector<Case> cases = {
        {1000, 1020, 500, 1600, 100, 6},
        {1000, 1020, 980, 1050, 100, 6},   // window too narrow
        {1000, 1020, 500, 1600, 250, 3},
        {-1, 20, 0, 100, 10, 4},           // invalid
        {1000, 1000, 0, 5000, 37, 5},      // zero width
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const auto controls = placer::make_breakpoint_shift_controls(
            c.bl, c.br, c.ws, c.we, c.step, c.maxc);
        ind();
        std::printf("{\"bp_left\": %d, \"bp_right\": %d, \"window_start\": %d, "
                    "\"window_end\": %d, \"step\": %d, \"max\": %d, "
                    "\"controls\": [", c.bl, c.br, c.ws, c.we, c.step, c.maxc);
        for (size_t j = 0; j < controls.size(); ++j) {
            std::printf("%s[%d, %d]", j ? ", " : "", controls[j].bp_left,
                        controls[j].bp_right);
        }
        std::printf("]}%s\n", i + 1 < cases.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");

    ind();
    std::printf("\"empirical_null_tail\": [\n");
    ++g_indent;
    placer::EmpiricalNullTail tail;
    for (int i = 0; i < 20; ++i) {
        tail.add(static_cast<double>(i));
    }
    const std::vector<double> probes = {-1.0, 0.0, 5.0, 10.0, 19.0, 20.0};
    for (size_t i = 0; i < probes.size(); ++i) {
        ind();
        std::printf("{\"observed\": %.17g, \"upper_tail_p\": %.17g, "
                    "\"n\": %d}%s\n", probes[i], tail.upper_tail_p(probes[i]),
                    static_cast<int>(tail.size()),
                    i + 1 < probes.size() ? "," : "");
    }
    --g_indent;
    ind();
    std::printf("],\n");
}

}  // namespace

int main() {
    std::printf("{\n");
    ++g_indent;
    ind();
    std::printf("\"_comment\": \"Golden vectors from the C++ decision layer. "
                "Regenerate with PLACER_py/tools/dump_oracle.cpp whenever the "
                "C++ changes on purpose; an unintended diff here means the "
                "migration contract moved.\",\n");
    dump_certificates();
    dump_genotypes();
    dump_overdispersion();
    dump_dependency_penalty();
    dump_event_explanations();
    dump_conformal();
    dump_null_controls();
    dump_sequence_structure();
    --g_indent;
    std::printf("}\n");
    return 0;
}
