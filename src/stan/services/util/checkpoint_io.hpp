#ifndef STAN_SERVICES_UTIL_CHECKPOINT_IO_HPP
#define STAN_SERVICES_UTIL_CHECKPOINT_IO_HPP

#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace util {

struct checkpoint_state {
  int version = 1;
  int chain_id = 1;
  std::string phase = "sampling";
  int iteration = 0;
  int num_warmup = 0;
  int num_samples = 0;
  int thin = 1;
  unsigned int random_seed = 0;
  std::string metric_type = "diag_e";
  double step_size = 1.0;
  std::vector<double> inv_metric;
  std::vector<double> last_position;
  double last_lp = 0.0;
  double last_accept_stat = 0.0;
  std::string rng_state;
};

/**
 * Options for sampling-phase checkpointing. CmdStan loads checkpoint files
 * and passes resume state; Stan writes checkpoints via the writer callback.
 */
struct checkpoint_options {
  int freq = 0;
  callbacks::structured_writer* writer = nullptr;
  const checkpoint_state* resume = nullptr;
  bool skip_headers = false;
};

namespace internal {

using noop_ostream_deleter = void (*)(std::ostream*);

inline void write_checkpoint_json(const checkpoint_state& state,
                                  std::ostream& out) {
  out.precision(std::numeric_limits<double>::max_digits10);
  stan::callbacks::json_writer<std::ostream, noop_ostream_deleter> writer(
      std::unique_ptr<std::ostream, noop_ostream_deleter>(
          &out, [](std::ostream*) {}));

  writer.begin_record();
  writer.write("version", state.version);
  writer.write("chain_id", state.chain_id);
  writer.write("phase", state.phase);
  writer.write("iteration", state.iteration);
  writer.write("num_warmup", state.num_warmup);
  writer.write("num_samples", state.num_samples);
  writer.write("thin", state.thin);
  writer.write("random_seed", static_cast<int>(state.random_seed));
  writer.write("metric_type", state.metric_type);
  writer.write("step_size", state.step_size);
  writer.write("inv_metric", state.inv_metric);
  writer.write("last_position", state.last_position);
  writer.write("last_lp", state.last_lp);
  writer.write("last_accept_stat", state.last_accept_stat);
  writer.write("rng_state", state.rng_state);
  writer.end_record();
}

inline std::vector<double> read_double_array(const rapidjson::Value& value,
                                             const char* name) {
  if (!value.HasMember(name) || !value[name].IsArray()) {
    throw std::runtime_error(std::string("checkpoint missing array: ") + name);
  }
  std::vector<double> result;
  result.reserve(value[name].Size());
  for (rapidjson::SizeType i = 0; i < value[name].Size(); ++i) {
    if (!value[name][i].IsNumber()) {
      throw std::runtime_error(std::string("checkpoint invalid array element: ")
                               + name);
    }
    result.push_back(value[name][i].GetDouble());
  }
  return result;
}

inline int read_int_field(const rapidjson::Value& value, const char* name) {
  if (!value.HasMember(name) || !value[name].IsInt()) {
    throw std::runtime_error(std::string("checkpoint missing int field: ")
                             + name);
  }
  return value[name].GetInt();
}

inline std::string read_string_field(const rapidjson::Value& value,
                                     const char* name) {
  if (!value.HasMember(name) || !value[name].IsString()) {
    throw std::runtime_error(std::string("checkpoint missing string field: ")
                             + name);
  }
  return value[name].GetString();
}

inline double read_double_field(const rapidjson::Value& value,
                                const char* name) {
  if (!value.HasMember(name) || !value[name].IsNumber()) {
    throw std::runtime_error(std::string("checkpoint missing numeric field: ")
                             + name);
  }
  return value[name].GetDouble();
}

inline double read_double_field_or_default(const rapidjson::Value& value,
                                           const char* name,
                                           double default_value) {
  if (!value.HasMember(name) || !value[name].IsNumber()) {
    return default_value;
  }
  return value[name].GetDouble();
}

}  // namespace internal

inline checkpoint_state read_checkpoint_json(const std::string& json) {
  rapidjson::Document document;
  if (document.Parse<0>(json.c_str()).HasParseError()) {
    throw std::runtime_error("checkpoint JSON parse error");
  }
  if (!document.IsObject()) {
    throw std::runtime_error("checkpoint JSON must be an object");
  }

  checkpoint_state state;
  state.version = internal::read_int_field(document, "version");
  if (state.version != 1) {
    throw std::runtime_error("unsupported checkpoint version");
  }
  state.chain_id = internal::read_int_field(document, "chain_id");
  state.phase = internal::read_string_field(document, "phase");
  state.iteration = internal::read_int_field(document, "iteration");
  state.num_warmup = internal::read_int_field(document, "num_warmup");
  state.num_samples = internal::read_int_field(document, "num_samples");
  state.thin = internal::read_int_field(document, "thin");
  state.random_seed
      = static_cast<unsigned int>(internal::read_int_field(document, "random_seed"));
  state.metric_type = internal::read_string_field(document, "metric_type");
  state.step_size = internal::read_double_field(document, "step_size");
  state.inv_metric = internal::read_double_array(document, "inv_metric");
  state.last_position = internal::read_double_array(document, "last_position");
  state.last_lp = internal::read_double_field_or_default(document, "last_lp", 0.0);
  state.last_accept_stat
      = internal::read_double_field_or_default(document, "last_accept_stat", 0.0);
  state.rng_state = internal::read_string_field(document, "rng_state");
  return state;
}

inline std::string serialize_checkpoint(const checkpoint_state& state) {
  std::ostringstream out;
  internal::write_checkpoint_json(state, out);
  return out.str();
}

inline bool checkpoint_exists(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) {
    return false;
  }
  try {
    std::stringstream buffer;
    buffer << in.rdbuf();
    read_checkpoint_json(buffer.str());
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

inline void write_checkpoint_to_writer(const checkpoint_state& state,
                                       callbacks::structured_writer& writer) {
  writer.begin_record();
  writer.write("version", state.version);
  writer.write("chain_id", state.chain_id);
  writer.write("phase", state.phase);
  writer.write("iteration", state.iteration);
  writer.write("num_warmup", state.num_warmup);
  writer.write("num_samples", state.num_samples);
  writer.write("thin", state.thin);
  writer.write("random_seed", static_cast<int>(state.random_seed));
  writer.write("metric_type", state.metric_type);
  writer.write("step_size", state.step_size);
  writer.write("inv_metric", state.inv_metric);
  writer.write("last_position", state.last_position);
  writer.write("last_lp", state.last_lp);
  writer.write("last_accept_stat", state.last_accept_stat);
  writer.write("rng_state", state.rng_state);
  writer.end_record();
}

inline void write_checkpoint(const checkpoint_state& state,
                             const std::string& path) {
  const std::string tmp_path = path + ".tmp";
  {
    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out.good()) {
      throw std::runtime_error("cannot write checkpoint file: " + tmp_path);
    }
    internal::write_checkpoint_json(state, out);
    out.flush();
    if (!out.good()) {
      throw std::runtime_error("checkpoint write failed: " + tmp_path);
    }
  }
  if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
    throw std::runtime_error("checkpoint rename failed: " + path);
  }
}

inline checkpoint_state read_checkpoint(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) {
    throw std::runtime_error("cannot read checkpoint file: " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return read_checkpoint_json(buffer.str());
}

}  // namespace util
}  // namespace services
}  // namespace stan

#endif
