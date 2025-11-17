# **ML Add-on for PROS - BETA VERSION**

My goal with this project was to add **machine learning assistance** to a **VEX V5 robot using PROS**.
This add-on is **modular** and designed to **drop into an existing project** (e.g., LemLib).
It provides:

* a tiny on-device **MLP inference engine**,
* a **feature scaler**,
* an **online linear learner**,
* **streaming helpers** (circular buffer, EWMA), and
* a **TFLite Micro wrapper stub**.
* Pair this with my library if need be - https://github.com/IshanJakkulwar/IJ-Template


This README assumes you are **new to both ML and VEX programming**.
I’ll walk you through everything step by step.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Folder structure](#2-folder-structure)
3. [Quick smoke test](#3-quick-smoke-test)
4. [Data collection](#4-data-collection--how-to-record-training-data-from-the-robot)
5. [Scaling features](#5-scaling-features--why-and-how)
6. [Training a model](#6-training-a-model--exporting-to-c-header)
7. [Deploying the model](#7-deploying-the-model-on-the-robot)
8. [Integration example (LemLib)](#8-example-integrate-with-lemlib-complete-copy-paste-example)
9. [Online adaptation](#9-online-adaptation--use-onlinelinear-for-small-on-robot-tuning)
10. [TFLite Micro](#10-tflite-micro-going-further--how-to-replace-the-stub)
11. [Troubleshooting](#11-troubleshooting--faqs)
12. [Safety notes](#12-safety-notes)
13. [Full LemLib example](#13-extra-full-lemlib-example-ready-to-copy)

---

## 1) Prerequisites

*  A computer with **Python 3.8+** and **pip**
*  **PROS V5 toolchain** installed and configured (`prosv5 make` should work)
*  (Optional but recommended) **VS Code** with the PROS extension
*  For training: **TensorFlow 2.x** (`pip install tensorflow`)

VS Code setup tip:
Add these paths in `.vscode/c_cpp_properties.json`:

```json
"includePath": [
    "${workspaceFolder}/include",
    "${env:HOME}/.pros/includes"
]
```

---

## 2) Folder structure

```
YOUR_PROS_PROJECT/
├─ include/
│  └─ ml_addon/
│     ├─ ml_addon.hpp       // main public header
│     ├─ tflm_wrapper.hpp   // TFLM stub
│     └─ nn_weights.h       // generated after training
├─ src/
│  ├─ ml_addon.cpp
│  └─ tflm_wrapper.cpp
├─ tools/
│  ├─ collect_data.py
│  ├─ scale_stats.py
│  └─ train_export.py
├─ examples/
│  └─ ml_residual_example.cpp
├─ .vscode/
│  └─ c_cpp_properties.json
└─ project.pros
```

 **PROS automatically includes `include/`**, so use:

```cpp
#include "ml_addon/ml_addon.hpp"
```

If IntelliSense fails in VS Code, reload the window after setting paths.

---

## 3) Quick smoke test

1. Put `ml_addon.hpp` in `include/ml_addon/`
2. Put `ml_addon.cpp` in `src/`
3. Keep `nn_weights.h` as a placeholder for now
4. Add the example file to `src/main.cpp`
5. Build with PROS:

   ```bash
   prosv5 make
   ```

If you see errors about `-std` or initializer lists, edit your Makefile:

```
CXXFLAGS += -std=gnu++17
```

---

## 4) Data collection — how to record training data from the robot

Collect (features → targets) data. Example CSV row:

```
target_rpm, battery_volts, measured_rpm, pose_x, pose_y, residual_left, residual_right
```

### Example logging snippet

```cpp
float target_rpm = ...;
float battery = pros::battery::get_capacity();
float measured_rpm = left_motor.get_actual_velocity();
float x = odom.getPose().x, y = odom.getPose().y;
float residual_l = known_residual_left;
float residual_r = known_residual_right;

printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
       target_rpm, battery, measured_rpm, x, y, residual_l, residual_r);
```

### Save from serial

```bash
python3 tools/collect_data.py /dev/tty.usbmodemXXXX 115200 out.csv
```

Tips:

* Record varied conditions (battery, load, terrain).
* Log both sides of drivetrain.

---

## 5) Scaling features

Neural nets train best when features share similar ranges.
Use `scale_stats.py`:

```bash
python3 tools/scale_stats.py out.csv out_dir --mode minmax
# or
python3 tools/scale_stats.py out.csv out_dir --mode standard
```

This produces:

```
scaler_minmax.h
```

Copy into `include/ml_addon/`.

Then in code:

```cpp
#include "scaler_minmax.h"
ml_addon::Scaler s;
s.load(SCALER_MIN, SCALER_MAX, ml_addon::Scaler::MINMAX);
auto scaled = s.transform(raw_features);
```

---

## 6) Training a model & exporting to C header

Use `train_export.py`:

```bash
python3 tools/train_export.py out.csv exported_model_dir
```

It will:

* Train a small **Keras MLP**
* Save:

  * `model.h5`
  * `model.tflite`
  * `nn_weights.h` (ready for C++ include)

Replace `include/ml_addon/nn_weights.h` in your PROS project.

---

## 7) Deploying the model on the robot

```cpp
#include "ml_addon/ml_addon.hpp"
#include "ml_addon/nn_weights.h"

ml_addon::MLP net;
net.load_from_arrays(ML_ADDON_W, ML_ADDON_B, ML_ADDON_SIZES, ML_ADDON_ACTS);

std::vector<float> input = scaled_features;
std::vector<float> out = net.infer(input);
```

Use `out` as a **feedforward residual** (start small, e.g., ×0.05).

---

## 8) Example: integrate with LemLib

Example showing ML residuals enhancing LemLib control:

```cpp
#include "pros/apix.h"
#include "ml_addon/ml_addon.hpp"
#include "ml_addon/nn_weights.h"
#include "scaler_minmax.h"

namespace lemlib {
    struct ChassisController {
        void set_target_velocity(float left, float right);
        void update();
    };
    extern ChassisController chassis;
}

class MLResidualWrapper {
public:
    MLResidualWrapper() {
        net_.load_from_arrays(ML_ADDON_W, ML_ADDON_B, ML_ADDON_SIZES, ML_ADDON_ACTS);
        scaler_.load(SCALER_MIN, SCALER_MAX, ml_addon::Scaler::MINMAX);
    }
    std::pair<float,float> get_residual(float target_rpm, float measured_rpm, float x, float y, float battery) {
        std::vector<float> raw = { target_rpm, battery, measured_rpm, x, y };
        auto inp = scaler_.transform(raw);
        auto out = net_.infer(inp);
        float ff_l = out.size() > 0 ? out[0] : 0.0f;
        float ff_r = out.size() > 1 ? out[1] : ff_l;
        return { 0.05f * ff_l, 0.05f * ff_r };
    }
private:
    ml_addon::MLP net_;
    ml_addon::Scaler scaler_;
};
```

---

## 9) Online adaptation (OnlineLinear)

```cpp
ml_addon::OnlineLinear online;
online.init(in_dim, out_dim, 1e-3f, 1e-5f);

auto pred = online.predict(features);
float corrected = base_target + 0.05f * pred[0];
online.update(features, {true_residual});
```

Use for **tiny on-robot learning** (safe adaptation).

---

## 10) TFLite Micro 

To run larger networks:

1. Convert model to `.tflite`
2. Add TFLite Micro sources to PROS project
3. Implement `tflm_wrapper.cpp` using:

   ```cpp
   tflite::GetModel();
   tflite::MicroInterpreter();
   ```
4. Tune memory arena (e.g., 64KB)

---

## 11) Troubleshooting & FAQs

| Problem           | Fix                                                     |
| ----------------- | ------------------------------------------------------- |
| `#include` errors | Add include paths in `.vscode/c_cpp_properties.json`    |
| Compile errors    | Add `CXXFLAGS += -std=gnu++17`                          |
| Model unstable    | Check feature order, scaling, and output scaling factor |
| Empty utils       | Fine — main logic is in `ml_addon.cpp/hpp`              |

---

## 12) Safety notes

⚠️ Always:

* Clamp ML outputs
* Start with small output multipliers (0.05)
* Test on bench before field use
* Log data for retraining

---

## 13) Full LemLib example (copy-ready)

```cpp
#include "pros/apix.h"
#include "ml_addon/ml_addon.hpp"
#include "ml_addon/nn_weights.h"
#include "scaler_minmax.h"

namespace lemlib {
    class ChassisController {
    public:
        void set_target_velocity(float l, float r) { left=l; right=r; }
        void update() {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "L:%.2f R:%.2f", left, right);
            pros::lcd::set_text(3, buf);
        }
    private: float left=0, right=0;
    };
    static ChassisController chassis;
}

class MLAdapter {
public:
    MLAdapter() {
        net_.load_from_arrays(ML_ADDON_W, ML_ADDON_B, ML_ADDON_SIZES, ML_ADDON_ACTS);
        s_.load(SCALER_MIN, SCALER_MAX, ml_addon::Scaler::MINMAX);
    }
    std::pair<float,float> predict_residual(float tgt, float meas, float x, float y, float batt) {
        std::vector<float> raw = { tgt, batt, meas, x, y };
        auto scaled = s_.transform(raw);
        auto out = net_.infer(scaled);
        float l = out.size()>0?out[0]:0, r = out.size()>1?out[1]:l;
        return {0.05f*l, 0.05f*r};
    }
private:
    ml_addon::MLP net_;
    ml_addon::Scaler s_;
};

MLAdapter gml;

int main() {
    pros::lcd::initialize();
    pros::lcd::set_text(1, "LemLib + IJ_ML Example");

    float base_rpm = 150, meas = 0, x=0, y=0;
    while (true) {
        pros::delay(10);
        meas += (rand()%3 - 1)*0.1f;
        float batt = pros::battery::get_capacity();
        auto [resL,resR] = gml.predict_residual(base_rpm, meas, x, y, batt);
        lemlib::chassis.set_target_velocity(base_rpm+resL, base_rpm+resR);
        lemlib::chassis.update();
    }
}
```

###  **LIBRARY FUNCTIONS**

Here’s what each component is for and what it enables in practice:

### 1) **Machine Learning Layer (ml_model.hpp / ml_model.cpp)**

* Defines and runs a **small neural network** (e.g., multilayer perceptron).
* You can train this network on your robot’s **telemetry data** (sensor readings, motor outputs, errors).
* Once trained, it can:

  * Predict corrections for drift or non-linear motor responses.
  * Adjust PID constants dynamically based on environment or battery voltage.
  * Predict position errors in odometry and compensate.

> Example: After driving 100 times, the robot “learns” how much it drifts left when turning and auto-corrects future paths.

---

### 2) **Dataset Management (data_handler.hpp / data_handler.cpp)**

* Logs input-output pairs during operation: e.g.,
  `(velocity_target, actual_velocity, error, output)`
* Saves datasets locally or streams via serial.
* Can later replay this data to retrain your model offline (Python/Colab).


---

### 3) **Auto-Tuning Interface (auto_tuner.hpp / auto_tuner.cpp)**

* Uses reinforcement learning or gradient search to auto-adjust PID gains.
* Can run between matches or at startup to calibrate automatically.
* You can choose strategies like:

  * Gradient-based fine-tuning.
  * Randomized exploration (e.g., try small PID variations).
  * Neural prediction of best PID set.

> Example: Before a match, robot runs a short self-test and tunes PID constants for optimal turning accuracy.

---

### 4) **Integration Layer (ml_addon.hpp / ml_addon.cpp)**

* Provides a simple interface for use in your main code:

  ```cpp
  ml_addon::AutoTuner tuner;
  ml_addon::MLModel model;

  tuner.runAutoTune();
  float correction = model.predict({targetVel, actualVel});
  ```
* Can attach to LemLib’s drive or odometry objects directly.

> So instead of rewriting your entire system, you “plug in” the ML layer to enhance it.


##  TL;DR summary

| Feature             | Description               | Outcome                          |
| ------------------- | ------------------------- | -------------------------------- |
| **Data Logger**     | Records robot performance | Lets you train ML models later   |
| **ML Model (NN)**   | Learns robot’s behavior   | Predicts corrections for control |
| **Auto-Tuner**      | Optimizes PID constants   | Improves accuracy automatically  |
| **Integration API** | Plug-and-play for LemLib  | No need to change main code      |

---

## Why it’s valuable

This addon bridges **classical control (PID, odometry)** and **machine learning (adaptive tuning, prediction)** which is a first of its kind for VEX-level robotics.

It can:

* Automatically adjust to friction, voltage, or mechanical wear.
* Adapt drive constants during long competitions.
* Predict motion errors and compensate in real time.
