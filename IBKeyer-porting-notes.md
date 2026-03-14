# IBKeyer Porting Notes


I tried to port this in a way that keeps the spirit of your original `IBKeyer` intact, while also making it survive the things OFX hosts do that a single-platform plugin can get away with ignoring for a while.

The big change is that I stopped treating the host glue, the algorithm, and the GPU backend as one blended piece of code. That older shape is convenient when you are moving fast, but it becomes hard to reason about once you need Windows, Linux, CUDA, Metal, CPU fallback, and host-specific safety checks all at once. So i split the work into:

- `IBKeyer.cpp` for OFX lifecycle and parameter wiring
- `IBKeyerBackend.cpp` for backend choice, packing, fallback, and CPU reference behavior
- `IBKeyerCuda.cu` for CUDA execution

I tried to leave comments in the moved code that say what used to live where.

## Backend policy

- On macOS, I keep the existing host-Metal path when the host gives us the Metal command queue and the old assumptions still hold.
- On Windows and Linux, I first try host CUDA zero-copy when the host explicitly enables CUDA render and gives us a CUDA stream.
- If the host CUDA contract is missing or not safe to trust for a given render, I fall back to the plugin's internal staged CUDA path.
- If CUDA is unavailable or fails, I fall back to CPU.

That means the practical order is:

- macOS: `Host Metal -> CPU`
- Windows/Linux: `Host CUDA -> Internal CUDA -> CPU`

That needs one important footnote now, because this was one of the bugs I had to fix after real Resolve testing:

- once the host has already switched the frame over to OFX host-CUDA memory, I do **not** let the plugin "fall back" into CPU staging just because the host-CUDA path declined

At that point the images are no longer normal host-readable pointers, so pretending staged CUDA or CPU is still a safe fallback is how you end up hanging the loader or scanning the plugin against the wrong memory model. So the real rule is:

- if the host never enabled CUDA render for that frame, `Internal CUDA -> CPU` is still a normal fallback chain
- if the host **did** enable CUDA render, IBKeyer stays on the host-CUDA memory contract and fails fast instead of trying to reinterpret those pointers on the CPU

I kept CPU as the reference implementation. It is slower, but it is also the least dependent on host resource ownership rules, stream lifetime, or GPU pointer assumptions. When I need to answer "is the algorithm wrong, or is the backend wrong?", CPU is the thing I compare against first.

## Why I split the old file

The original layout was fine for getting the effect working, but it mixed a few different kinds of logic:

- OFX clip/parameter fetching
- backend selection
- pixel packing assumptions
- algorithm math
- GPU dispatch details

The real risk is silent divergence, where each backend slowly grows its own assumptions about bounds, origin, row stride, or clip layout.

So the split is mostly about making the contracts explicit:

- `IBKeyer.cpp` asks the host for images and render-state
- `IBKeyerBackend.cpp` translates that into a stable internal request
- the backend files do the actual processing


## Why rowBytes and bounds had to become first-class

It's understandable to assume that `getPixelData()` behaves like a tightly packed RGBA frame that starts at `(0, 0)`.

Sometimes it does.

Sometimes the host gives you padded rows, cropped windows, non-zero bounds, or a resource that only makes sense if you compute addresses through the host's layout rules. That is where the old "just treat the pointer like a flat image" style starts drifitng.

So I introduced image descriptors that carry:

- base pointer
- bounds
- rowBytes
- component count

That sounds more verbose than it used to be, but it makes a difference between "works on my test clip" and "works when Resolve hands us a partial render window at a non-zero origin."

## Why there are still two CUDA paths

I know this looks redundant at first glance.

The host CUDA path is the fast path. It is there to avoid copying source and destination pixels through CPU memory when the host is already willing to let the plugin render directly on device memory. That is the zero-copy path.

But rhe way I learned to do it is by no leting the plugin become dependent on that path existing in every host, or existing safely in every render situation. So I kept the internal staged CUDA path too. That path:

- packs the requested render window into plugin-owned contiguous buffers
- uploads those buffers to CUDA
- runs the kernels
- copies results back out

The staged path is not the dream path for performance, but it is a very useful safety net. It lets Windows and Linux keep using CUDA even if the host does not expose the OFX CUDA interop contract we need, or if a specific render has layout details that make zero-copy too risky to trust.

TLDR:

- host CUDA exists for performance
- staged CUDA exists for portability and fallback safety

## Why zero-copy still needs scratch buffers

This is one of those things that looks like it should not be necessary until you try to write the kernels. even when the source and destination stay on host-owned device memory, the guided filter still needs intermediate images such as:

- raw alpha
- guide
- means and covariance terms
- scratch buffers for the separable Gaussian passes

Those intermediates are temporary working memory for the algorithm itself, so the plugin still has to allocate and manage them on the CUDA side.

That is why "zero-copy" here means "no CPU staging for source and destination on the fast path," not "no allocations anywhere."

## Why CPU is still the truth anchor

I resisted the temptation to make GPU behavior the definition of correctness.

The reason is simple: backend code is where host assumptions, stream handling, and memory layout bugs like to hide. The CPU is easier to inspect and less entangled with host-specific interop.

So when a render looks wrong, the intended debugging order is:

1. compare against CPU
2. compare staged CUDA against CPU
3. compare host CUDA against staged CUDA and CPU

That ordering makes it easier to tell whether a mismatch is caused by the algorithm itself, by the CUDA implementation, or by the host-resource contract.

## Why the advertised GPU support is narrower than the old file suggested

The innitial version advertised more GPU support than the implementation could safely honor across platforms.

I tried to make the descriptor honest.

- macOS advertises Metal because there is still a real host-Metal implementation there
- Windows/Linux advertise CUDA render and CUDA stream support because there is now a real host-CUDA path
- the plugin still keeps internal CUDA for fallback, but that is not the same thing as claiming every OFX GPU contract is implemented on every platform

## Fallback rules I kept explicit

I made the fallback policy noisy on purpose, because backend bugs are much easier to diagnose when the plugin says what it chose and why.

Useful environment switches:

- `IBKEYER_FORCE_CPU=1`
  Forces the reference CPU path.
- `IBKEYER_DISABLE_CUDA=1`
  Disables the internal staged CUDA path.
- `IBKEYER_CUDA_RENDER_MODE=HOST|AUTO|INTERNAL`
  Lets me force host-preferred or internal-only CUDA policy without changing the descriptor code again.
- `IBKEYER_DEBUG_LOG=1`
  Enables verbose backend logging.
- `IBKEYER_FILE_LOG=1`
  Writes those backend messages to a dedicated log file, which is much easier to read than chasing Resolve helper-process output.
- `IBKEYER_LOG_PATH=...`
  Optional override for where that file log is written.
- `IBKEYER_HOST_CUDA_FORCE_SYNC=1`
  Forces synchronization in the host-CUDA path when debugging stream/lifetime issues.

Normal fallback behavior:

- Host CUDA falls back to internal staged CUDA only when the host never actually put the frame on the OFX host-CUDA contract for that render.
- If the host **did** enable CUDA render and the host-CUDA path fails, I now stop there instead of falling through into CPU-readable staging assumptions.
- Internal CUDA falls back if staging, allocation, launch, or synchronization fails.
- Host Metal falls back when the host does not provide the expected full-frame resource layout.

That last point matters because I would rather lose performance than silently render against the wrong memory interpretation.

## Known risks 

The areas I would still validate manually are:

- Resolve on Windows and Linux, especially to confirm the host CUDA path is really entering zero-copy mode on live renders
- non-zero bounds and cropped render windows
- optional Screen clip in both RGB and RGBA forms
- parity between CPU, staged CUDA, host CUDA, and Metal on the same shot
- guided-filter edge detail and near-grey extraction behavior on real keyed footage

I also intentionally kept the macOS Metal path conservative. If the host does not give the old full-frame assumptions the Metal kernel expects, I would rather fall back to CPU than pretend partial-window Metal is safe when we have not proven it is.

One other Windows-specific lesson that turned out to matter more than I expected: loader safety is part of backend design too. I originally had a `thread_local` CUDA scratch cache with a destructor that cleaned up CUDA allocations and events. That looked tidy in isolation, but it is exactly the wrong kind of cleanup to run while `OFXLoader.exe` is unloading the plugin or tearing down threads. I changed that into a lazily allocated per-thread cache without destructor-driven CUDA teardown, because the boring version there is much safer than the elegant one.
