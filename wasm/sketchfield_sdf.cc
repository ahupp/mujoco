// Copyright 2026 Sketchfield contributors. Licensed under Apache-2.0.
// Synchronous, module-local SDF callbacks. Handles are owned by the JS worker;
// mjData holds only an immutable numeric handle, never a borrowed JS pointer.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <mujoco/mujoco.h>

EM_JS(double, SketchfieldDistance, (int field, double x, double y, double z), {
  return Module.sketchfieldDistance(field, x, y, z);
});
EM_JS(void, SketchfieldGradient, (int field, double x, double y, double z, double* out), {
  const value = Module.sketchfieldGradient(field, x, y, z);
  HEAPF64[out / 8] = value[0];
  HEAPF64[out / 8 + 1] = value[1];
  HEAPF64[out / 8 + 2] = value[2];
});
namespace {
const char* attributes[] = {"field", "cx", "cy", "cz", "hx", "hy", "hz"};
void ParseAttributes(mjtNum* out, const char** names, const char** values) {
  for (int i = 0; i < 7; ++i) {
    out[i] = NAN;
    for (int j = 0; j < 7; ++j) {
      if (!std::strcmp(names[j], attributes[i])) {
        char* end;
        double value = std::strtod(values[j], &end);
        if (end != values[j] && !*end && std::isfinite(value)) out[i] = value;
      }
    }
  }
}
void Register() {
  mjpPlugin p;
  mjp_defaultPlugin(&p);
  p.name = "sketchfield.sdf";
  p.capabilityflags = mjPLUGIN_SDF;
  p.nattribute = 7;
  p.attributes = attributes;
  p.nstate = +[](const mjModel*, int) { return 0; };
  p.init = +[](const mjModel* m, mjData* d, int i) {
    const char* value = mj_getPluginConfig(m, i, "field");
    char* end;
    double field = std::strtod(value, &end);
    if (end == value || *end || !std::isfinite(field) || field < 0 || field > 2147483646 || field != std::floor(field)) return -1;
    d->plugin_data[i] = static_cast<uintptr_t>(field) + 1;
    return 0;
  };
  p.reset = +[](const mjModel*, mjtNum*, void*, int) {};
  p.compute = +[](const mjModel*, mjData*, int, int) {};
  p.sdf_distance = +[](const mjtNum* x, const mjData* d, int i) {
    return SketchfieldDistance(static_cast<int>(d->plugin_data[i] - 1), x[0], x[1], x[2]);
  };
  p.sdf_gradient = +[](mjtNum* g, const mjtNum* x, const mjData* d, int i) {
    SketchfieldGradient(static_cast<int>(d->plugin_data[i] - 1), x[0], x[1], x[2], g);
  };
  p.sdf_staticdistance = +[](const mjtNum* x, const mjtNum* a) { return SketchfieldDistance(static_cast<int>(a[0]), x[0], x[1], x[2]); };
  p.sdf_attribute = ParseAttributes;
  p.sdf_aabb = +[](mjtNum* out, const mjtNum* a) { for (int i = 0; i < 6; ++i) out[i] = a[i + 1]; };
  mjp_registerPlugin(&p);
}
}
EMSCRIPTEN_BINDINGS(sketchfield_sdf) { Register(); }
