#ifndef STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/services/util/checkpoint_io.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <tbb/parallel_for.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

namespace stan {
namespace services {
namespace util {

namespace internal {

template <typename Sampler>
using sampler_point_type
    = std::decay_t<decltype(std::declval<Sampler&>().z())>;

template <typename Sampler>
constexpr bool has_diag_e_point
    = std::is_same_v<sampler_point_type<Sampler>, stan::mcmc::diag_e_point>;

template <typename Sampler>
constexpr bool has_dense_e_point
    = std::is_same_v<sampler_point_type<Sampler>, stan::mcmc::dense_e_point>;

template <typename Sampler>
void restore_sampler_metric(Sampler& sampler, const checkpoint_state& state) {
  if (state.metric_type == "diag_e") {
    if constexpr (has_diag_e_point<Sampler>) {
      Eigen::VectorXd vec(state.inv_metric.size());
      Eigen::Map<const Eigen::VectorXd> src(state.inv_metric.data(),
                                            state.inv_metric.size());
      vec = src;
      sampler.set_metric(vec);
    }
  } else if (state.metric_type == "dense_e") {
    if constexpr (has_dense_e_point<Sampler>) {
      const int n = static_cast<int>(state.inv_metric.size());
      const int d = static_cast<int>(std::sqrt(static_cast<double>(n)));
      Eigen::MatrixXd mat(d, d);
      Eigen::Map<const Eigen::MatrixXd> src(state.inv_metric.data(), d, d);
      mat = src;
      sampler.set_metric(mat);
    }
  }
}

template <typename Sampler, typename RNG>
checkpoint_state build_checkpoint_state(const Sampler& sampler,
                                        const stan::mcmc::sample& sample,
                                        RNG& rng, int absolute_iteration,
                                        int num_warmup, int num_samples,
                                        int num_thin, unsigned int random_seed,
                                        size_t chain_id) {
  checkpoint_state state;
  state.chain_id = static_cast<int>(chain_id);
  state.phase = "sampling";
  state.iteration = absolute_iteration;
  state.num_warmup = num_warmup;
  state.num_samples = num_samples;
  state.thin = num_thin;
  state.random_seed = random_seed;
  state.metric_type = sampler.get_metric_type();
  state.step_size = sampler.get_nominal_stepsize();
  state.inv_metric = sampler.get_inv_metric();
  state.last_position.assign(sample.cont_params().data(),
                             sample.cont_params().data()
                                 + sample.cont_params().size());
  state.last_lp = sample.log_prob();
  state.last_accept_stat = sample.accept_stat();
  std::ostringstream oss;
  oss << rng;
  state.rng_state = oss.str();
  return state;
}

}  // namespace internal

/**
 * Runs the sampler with adaptation, with writers for the sample,
 * diagnostics, and the adapted hmc tuning parameters.
 *
 * @tparam Sampler Type of adaptive sampler.
 * @tparam Model Type of model
 * @tparam RNG Type of random number generator
 * @param[in,out] sampler the mcmc sampler to use on the model
 * @param[in] model the model concept to use for computing log probability
 * @param[in] cont_vector initial parameter values
 * @param[in] num_warmup number of warmup draws
 * @param[in] num_samples number of post warmup draws
 * @param[in] num_thin number to thin the draws. Must be greater than
 *   or equal to 1.
 * @param[in] refresh controls output to the <code>logger</code>
 * @param[in] save_warmup indicates whether the warmup draws should be
 *   sent to the sample writer
 * @param[in,out] rng random number generator
 * @param[in,out] interrupt interrupt callback
 * @param[in,out] logger logger for messages
 * @param[in,out] sample_writer writer for draws
 * @param[in,out] diagnostic_writer writer for diagnostic information
 * @param[in,out] metric_writer writer for adapted stepsize, metric
 * @param[in] chain_id The id for a given chain, (optional, default == 1)
 * @param[in] num_chains The number of chains used in the program. This
 *  is used in generate transitions to print out the chain number,
 *  (optional, default == 1)
 * @param[in] random_seed random seed used for the run (for checkpointing)
 * @param[in] checkpoint checkpoint options including resume state and writer
 */
template <typename Sampler, typename Model, typename RNG>
void run_adaptive_sampler(Sampler& sampler, Model& model,
                          std::vector<double>& cont_vector, int num_warmup,
                          int num_samples, int num_thin, int refresh,
                          bool save_warmup, RNG& rng,
                          callbacks::interrupt& interrupt,
                          callbacks::logger& logger,
                          callbacks::writer& sample_writer,
                          callbacks::writer& diagnostic_writer,
                          callbacks::structured_writer& metric_writer,
                          size_t chain_id = 1, size_t num_chains = 1,
                          unsigned int random_seed = 0,
                          const checkpoint_options& checkpoint
                          = checkpoint_options{}) {
  const int finish = num_warmup + num_samples;
  services::util::mcmc_writer writer(sample_writer, diagnostic_writer, logger);

  checkpoint_save_callback save_checkpoint;
  if (checkpoint.freq > 0 && checkpoint.writer != nullptr) {
    save_checkpoint = [&](int absolute_iteration, stan::mcmc::sample& sample) {
      checkpoint_state state = internal::build_checkpoint_state(
          sampler, sample, rng, absolute_iteration, num_warmup, num_samples,
          num_thin, random_seed, chain_id);
      write_checkpoint_to_writer(state, *checkpoint.writer);
    };
  }

  if (checkpoint.resume != nullptr) {
    const checkpoint_state& state = *checkpoint.resume;
    std::istringstream iss(state.rng_state);
    iss >> rng;
    if (iss.fail()) {
      logger.error("Failed to restore RNG state from checkpoint.");
      return;
    }
    cont_vector = state.last_position;
    internal::restore_sampler_metric(sampler, state);
    sampler.disengage_adaptation();
    sampler.set_nominal_stepsize(state.step_size);

    Eigen::Map<Eigen::VectorXd> cont_params(cont_vector.data(),
                                            cont_vector.size());
    sampler.seed(cont_params);

    stan::mcmc::sample s(cont_params, state.last_lp, state.last_accept_stat);
    if (!checkpoint.skip_headers) {
      writer.write_sample_names(s, sampler, model);
      writer.write_diagnostic_names(s, sampler, model);
    }

    const int remaining = finish - state.iteration;
    auto start_sample = std::chrono::steady_clock::now();
    util::generate_transitions(sampler, remaining, state.iteration, finish,
                               num_thin, refresh, true, false, writer, s, model,
                               rng, interrupt, logger, chain_id, num_chains,
                               num_warmup, checkpoint.freq, save_checkpoint);
    auto end_sample = std::chrono::steady_clock::now();
    double sample_delta_t = std::chrono::duration_cast<std::chrono::milliseconds>(
                                end_sample - start_sample)
                                .count()
                            / 1000.0;
    writer.write_timing(0.0, sample_delta_t);
    return;
  }

  Eigen::Map<Eigen::VectorXd> cont_params(cont_vector.data(),
                                          cont_vector.size());
  sampler.engage_adaptation();
  try {
    sampler.z().q = cont_params;
    sampler.init_stepsize(logger);
  } catch (const std::exception& e) {
    logger.error("Exception initializing step size.");
    logger.error(e.what());
    return;
  }

  stan::mcmc::sample s(cont_params, 0, 0);

  // Headers
  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  auto start_warm = std::chrono::steady_clock::now();
  util::generate_transitions(sampler, num_warmup, 0, finish, num_thin, refresh,
                             save_warmup, true, writer, s, model, rng,
                             interrupt, logger, chain_id, num_chains);
  auto end_warm = std::chrono::steady_clock::now();
  double warm_delta_t = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_warm - start_warm)
                            .count()
                        / 1000.0;
  sampler.disengage_adaptation();
  writer.write_adapt_finish(sampler);
  sampler.write_sampler_state(sample_writer);
  sampler.write_sampler_state_struct(metric_writer);

  auto start_sample = std::chrono::steady_clock::now();
  util::generate_transitions(sampler, num_samples, num_warmup, finish, num_thin,
                             refresh, true, false, writer, s, model, rng,
                             interrupt, logger, chain_id, num_chains,
                             num_warmup, checkpoint.freq, save_checkpoint);
  auto end_sample = std::chrono::steady_clock::now();
  double sample_delta_t = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_sample - start_sample)
                              .count()
                          / 1000.0;
  writer.write_timing(warm_delta_t, sample_delta_t);
}

/**
 * Runs the sampler with adaptation.
 *
 * @tparam Sampler Type of adaptive sampler.
 * @tparam Model Type of model
 * @tparam RNG Type of random number generator
 * @param[in,out] sampler the mcmc sampler to use on the model
 * @param[in] model the model concept to use for computing log probability
 * @param[in] cont_vector initial parameter values
 * @param[in] num_warmup number of warmup draws
 * @param[in] num_samples number of post warmup draws
 * @param[in] num_thin number to thin the draws. Must be greater than
 *   or equal to 1.
 * @param[in] refresh controls output to the <code>logger</code>
 * @param[in] save_warmup indicates whether the warmup draws should be
 *   sent to the sample writer
 * @param[in,out] rng random number generator
 * @param[in,out] interrupt interrupt callback
 * @param[in,out] logger logger for messages
 * @param[in,out] sample_writer writer for draws
 * @param[in,out] diagnostic_writer writer for diagnostic information
 * @param[in] chain_id The id for a given chain, (optional, default == 1)
 * @param[in] num_chains The number of chains used in the program. This
 *  is used in generate transitions to print out the chain number,
 *  (optional, default == 1)
 */
template <typename Sampler, typename Model, typename RNG>
void run_adaptive_sampler(Sampler& sampler, Model& model,
                          std::vector<double>& cont_vector, int num_warmup,
                          int num_samples, int num_thin, int refresh,
                          bool save_warmup, RNG& rng,
                          callbacks::interrupt& interrupt,
                          callbacks::logger& logger,
                          callbacks::writer& sample_writer,
                          callbacks::writer& diagnostic_writer,
                          size_t chain_id = 1, size_t num_chains = 1) {
  callbacks::structured_writer dummy_metric_writer;
  return run_adaptive_sampler(
      sampler, model, cont_vector, num_warmup, num_samples, num_thin, refresh,
      save_warmup, rng, interrupt, logger, sample_writer, diagnostic_writer,
      dummy_metric_writer, chain_id, num_chains);
}

}  // namespace util
}  // namespace services
}  // namespace stan
#endif
