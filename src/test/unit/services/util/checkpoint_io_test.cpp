#include <stan/services/util/checkpoint_io.hpp>
#include <stan/services/util/create_rng.hpp>
#include <gtest/gtest.h>
#include <fstream>

TEST(checkpoint_io, round_trip_serialization) {
  stan::services::util::checkpoint_state state;
  state.version = 1;
  state.chain_id = 2;
  state.phase = "sampling";
  state.iteration = 500;
  state.num_warmup = 200;
  state.num_samples = 800;
  state.thin = 1;
  state.random_seed = 1234;
  state.metric_type = "diag_e";
  state.step_size = 0.412;
  state.inv_metric = {0.81, 1.24, 0.97, 1.01, 0.88, 0.92, 1.05, 0.79, 0.86,
                      1.11};
  state.last_position = {0.12, -0.33, 1.06, 0.44, -0.21, 0.05, 0.88, -0.17,
                         0.31, 0.62};
  state.last_lp = -12.34;
  state.last_accept_stat = 0.87;

  stan::rng_t rng = stan::services::util::create_rng(1234, 2);
  rng();
  std::ostringstream oss;
  oss << rng;
  state.rng_state = oss.str();

  std::string json = stan::services::util::serialize_checkpoint(state);
  stan::services::util::checkpoint_state restored
      = stan::services::util::read_checkpoint_json(json);

  EXPECT_EQ(state.version, restored.version);
  EXPECT_EQ(state.chain_id, restored.chain_id);
  EXPECT_EQ(state.phase, restored.phase);
  EXPECT_EQ(state.iteration, restored.iteration);
  EXPECT_EQ(state.num_warmup, restored.num_warmup);
  EXPECT_EQ(state.num_samples, restored.num_samples);
  EXPECT_EQ(state.thin, restored.thin);
  EXPECT_EQ(state.random_seed, restored.random_seed);
  EXPECT_EQ(state.metric_type, restored.metric_type);
  EXPECT_DOUBLE_EQ(state.step_size, restored.step_size);
  EXPECT_EQ(state.inv_metric, restored.inv_metric);
  EXPECT_EQ(state.last_position, restored.last_position);
  EXPECT_DOUBLE_EQ(state.last_lp, restored.last_lp);
  EXPECT_DOUBLE_EQ(state.last_accept_stat, restored.last_accept_stat);
  EXPECT_EQ(state.rng_state, restored.rng_state);

  stan::rng_t rng_restored = stan::services::util::create_rng(0, 1);
  std::istringstream iss(restored.rng_state);
  iss >> rng_restored;
  EXPECT_EQ(rng, rng_restored);
}

TEST(checkpoint_io, atomic_write_and_read) {
  stan::services::util::checkpoint_state state;
  state.iteration = 300;
  state.num_warmup = 200;
  state.num_samples = 800;
  state.thin = 1;
  state.random_seed = 1234;
  state.inv_metric = {1.0, 2.0};
  state.last_position = {0.1, 0.2};
  state.rng_state = "0";

  const std::string path = "test_checkpoint_io.json";
  stan::services::util::write_checkpoint(state, path);
  EXPECT_TRUE(stan::services::util::checkpoint_exists(path));

  stan::services::util::checkpoint_state loaded
      = stan::services::util::read_checkpoint(path);
  EXPECT_EQ(state.iteration, loaded.iteration);
  EXPECT_EQ(state.inv_metric, loaded.inv_metric);

  std::remove(path.c_str());
}

TEST(checkpoint_io, missing_file_throws) {
  EXPECT_THROW(stan::services::util::read_checkpoint("missing_checkpoint.json"),
               std::runtime_error);
}

TEST(checkpoint_io, truncated_json_throws) {
  EXPECT_THROW(stan::services::util::read_checkpoint_json("{ \"version\": 1"),
               std::runtime_error);
}

TEST(checkpoint_io, fixture_rng_state_restores) {
  const std::string path
      = "src/test/interface/fixtures/checkpoint/checkpoint_at_500.json";
  std::ifstream in(path);
  if (!in.good()) {
    GTEST_SKIP() << "checkpoint fixture not generated";
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  stan::services::util::checkpoint_state state
      = stan::services::util::read_checkpoint_json(buffer.str());

  stan::rng_t rng = stan::services::util::create_rng(state.random_seed,
                                                     state.chain_id);
  std::istringstream iss(state.rng_state);
  iss >> rng;
  ASSERT_FALSE(iss.fail());

  std::ostringstream oss;
  oss << rng;
  EXPECT_EQ(state.rng_state, oss.str());
}
