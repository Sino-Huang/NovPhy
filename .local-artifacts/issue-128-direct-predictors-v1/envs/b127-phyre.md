# Conda env `b127-phyre` (#128 A1/A2: official PHYRE simulator + DQN agents)

Built 2026-10-24. Package list: `envs/b127-phyre.txt` (`conda list --explicit` + `pip freeze`).

## Why not the pip wheel / env.yml
- PyPI `phyre` 0.2.2 ships manylinux wheels for cp36–cp39 only.
- The documented source recipes (env.yml: python 3.6, `thrift-cpp=0.11.0`, boost 1.67; INSTALLATION.md: python 3.9,
  `thrift-cpp=0.11.0`, boost 1.75, pybind11 2.6) no longer solve on conda-forge: every `thrift-cpp 0.11.0` build pins
  `openssl 1.0` / `zlib 1.2`, which conflict with every available python 3.9–3.11 build (`conda create --dry-run`,
  2026-10-24). So thrift 0.11.0 (the version the Python side pins, `thrift==0.11.0`) is built from source into the env.

## Recipe (Python 3.11, the newest the build supports cleanly: torch 2.13 / thrift 0.11 / pybind11 3)
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n b127-phyre -c conda-forge --override-channels python=3.11 sed nodejs cmake make pybind11 \
    libboost-devel setuptools pip wget pkg-config            # python 3.11.17, cmake 4.4.4, pybind11 3.1.0, boost 1.92.0
conda activate b127-phyre

# thrift 0.11.0 C++ library + compiler, static, PIC, into $CONDA_PREFIX
#   https://archive.apache.org/dist/thrift/0.11.0/thrift-0.11.0.tar.gz
#   sha256 c4ad38b6cb4a3498310d405a91fef37b9a8e79a50cd0968148ee2524d2fa60c2
cd /mnt/array/sukaih/novphy-baselines-src/_build && tar xzf thrift-0.11.0.tar.gz && mkdir thrift-0.11.0/cmb && cd thrift-0.11.0/cmb
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake .. -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_TUTORIALS=OFF -DBUILD_EXAMPLES=OFF -DWITH_SHARED_LIB=OFF \
  -DWITH_STATIC_LIB=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DWITH_OPENSSL=OFF -DWITH_ZLIB=OFF -DWITH_LIBEVENT=OFF \
  -DWITH_QT4=OFF -DWITH_QT5=OFF -DWITH_C_GLIB=OFF -DWITH_JAVA=OFF -DWITH_PYTHON=OFF -DWITH_HASKELL=OFF \
  -DWITH_STDTHREADS=ON -DBOOST_ROOT=$CONDA_PREFIX
make -j16 && make install                                     # -> $CONDA_PREFIX/lib/libthrift.a, bin/thrift (0.11.0)
# (C++14 because boost 1.92's headers need it; thrift 0.11 defaults to C++11.)

# PHYRE simulator from a build copy of the pinned clone (the read-only clone stays untouched)
git clone /mnt/array/sukaih/novphy-baselines-src/phyre@d0765dd /mnt/array/sukaih/novphy-baselines-src/_build/phyre@d0765dd
cd /mnt/array/sukaih/novphy-baselines-src/_build/phyre@d0765dd    # HEAD d0765dd2f6d6cf41b933a3a57173c63d21fbfa6a
git submodule update --init                                   # Box2D 33af62e, clip2tri f62a734, googletest 4e4df22
CMAKE_POLICY_VERSION_MINIMUM=3.5 CXXFLAGS="-include cstdio -include cstdint" make compile
#   (Box2D ConvexDecomposition uses printf without <cstdio>, an error under gcc 13; no source edits)
#   -> src/python/phyre/simulator_bindings.cpython-311-x86_64-linux-gnu.so, cmake_build/gen-py (thrift Python)
(cd src/python && python setup.py egg_info)
echo "$PWD/src/python" > $(python -c "import site;print(site.getsitepackages()[0])")/phyre-develop.pth
#   = what `pip install -e src/python` does, minus its `make react_deps` / web-viz build (npm; not needed)
pip install nose numpy tornado thrift==0.11.0 imageio scipy joblib tqdm Pillow matplotlib pytest
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130   # torch 2.13.0+cu130 (= novphy), torchvision 0.28.0
```
Runtime environment: `PHYRE_CACHE_DIR=/mnt/array/sukaih/Project/NovPhy/data/external/phyre_dqn/phyre_cache` (the
downloaded 100k cache), `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to load the 2020 DQN checkpoints with the official
`torch.load(path)` call.

## Verification (CPU, `CUDA_VISIBLE_DEVICES=""`, 2026-10-24)
- `import phyre` → the build copy; `phyre.MAIN_EVAL_SETUPS` lists the 5 setups; `get_fold('ball_within_template', 0)`
  → 1600 / 400 / 500 train / dev / test tasks; `initialize_simulator(..., 'ball')` → initial scenes (N, 256, 256),
  action dim 3; `simulate_action` and `Evaluator` run.
- 100k cache: `get_default_100k_cache('ball')` → 100 000 actions × 3, 3 700 tasks; 1 000 cached (task, action)
  statuses (20 fold-0 test tasks × 50 random actions) re-simulated with this build: **0 mismatches**.
- Official unit tests `python -m pytest phyre/tests` (src/python): **48 passed, 2 skipped, 2 failed**; the two failures
  are `server_test.py` (viz server) calling `scipy.stats.binom_test`, removed in scipy 1.12 (task-complexity stats
  of the web demo; not used by the agents, cache, folds or evaluator).
- Official agents: `agents/neural_agent.train` runs unmodified (G-port reference, 3 updates on PHYRE-B data);
  `agents/train.py --dqn-load-from <shipped ckpt>` evaluates (CPU smoke, 3 tasks, `model.cuda` patched to the
  identity for the smoke only).
