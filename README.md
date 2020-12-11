# Advent of GPU Code 2020

[![Twitch Badge](https://img.shields.io/badge/Twitch-ConstrainedCoding-9147ff?logo=twitch&logoColor=white)](https://www.twitch.tv/constrainedcoding)
[![Periscope Badge](https://img.shields.io/badge/Periscope-__JacobTomlinson-%2340A4C4?logo=periscope&logoColor=white)](https://www.pscp.tv/_JacobTomlinson/follow)
[![YouTube Badge](https://img.shields.io/badge/YouTube-Jacob%20Tomlinson-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCjwcSpcyRYsfZMsliAJzYuQ/live)

This repo contains solutions to the 2020 [Advent of Code](https://adventofcode.com/) written for the [GPU using Python](https://numba.pydata.org/numba-doc/dev/cuda/overview.html).

Solutions will be written live on [Twitch](https://www.twitch.tv/constrainedcoding) and then recordings will be published to [YouTube](https://www.youtube.com/channel/UCjwcSpcyRYsfZMsliAJzYuQ) later.

## FAQ

### General

#### What is Advent of Code?

[Advent of Code](https://adventofcode.com/) is a series of computer science problems released each day throughout December.

Participants have to write code to solve each problem.

#### Who are you?

My name is [Jacob Tomlinson](https://twitter.com/_jacobtomlinson). I work for NVIDIA maintaining open source Python projects including [Dask](https://dask.org/) and [RAPIDS](https://rapids.ai/).

#### You don't really seem to know what you are doing?

That's because I'm learning in public.

All of my work involves building distributed systems in Python. I focus heavily on deploying and building clusters of
machines to carry out data science workloads and then giving users visibility into what those clusters are doing. 

I am originally from a sysadmin background and am not a data scientist. Therefore doing the Advent of Code in Python on the GPU is a way for me to level up my GPU coding skills and bring you all along for the ride.

One of the Numba devs [Graham Markall](https://twitter.com/gmarkall) is also solving Advent of Code on the GPU, so check out [his solutions](https://github.com/gmarkall/advent-of-numba). He actually knows what he is doing!

#### What kind of GPU are you using?

I generally SSH to a machine with a pair of NVIDIA Quadro RTX 8000s. I also have a laptop with an NVIDIA GeForce GTX 1050.

If you want to try it for yourself you'll need a Pascal series GPU or better. Check out the RAPIDS list of [supported GPUs](https://medium.com/dropout-analytics/which-gpus-work-with-rapids-ai-f562ef29c75f).

#### Is there a better way to solve that problem?

Quite possibly!

My goal here is to increase my skill in GPU development in Python, rather than solve the problems perfectly. 

I also do not have a formal computer science education, so I find things like Advent of Code really useful for building my computer science fundamentals. 

If you want to give me pointers and tips in the live stream chat then please do!

#### Can everything be done on the GPU?

Not all problems map onto something that can run in parallel, therefore I am not expecting to solve every challenge on the GPU. Instead I am aiming to do *as much as possible* on the GPU. Code that can be parallelised on the GPU will be faster than it's CPU counterpart.

Here's a table of how often something could be parallelised on the GPU.

| Day | Part 1 | Part 2 |
| --- | ------ | ------ |
| 01 | ✅ | ✅ |
| 02 | ✅ | ✅ |
| 03 | ✅ | ✅ |
| 04 | ✅ | ✅ |
| 05 | ✅ | Potentially |
| 06 | ✅ | ✅ |
| 07 | ❌ | ❌ |
| 08 | ❌ | ✅ |
| 09 | ✅ | ✅ |
| 10 | ✅ | ❌ |


### Technical

#### Which Python libraries are you using?

I plan to mostly use [Numba's CUDA support](https://numba.pydata.org/numba-doc/dev/cuda/overview.html). This allows me to write Python and execute it on the GPU.

I may also use [CuPy](https://cupy.dev/), [cuDF](https://github.com/rapidsai/cudf) and other packages in the [RAPIDS ecosystem](https://github.com/rapidsai).

#### What Python environment are you using?

I am using the latest [RAPIDS Docker image](https://hub.docker.com/r/rapidsai/rapidsai/), which contains the RAPIDS packages plus some extras such as Jupyter Lab.

You can find instructions on installing RAPIDS via Docker or Conda [here](https://rapids.ai/start.html#get-rapids).

#### What is a kernel?

A CUDA kernel is a fancy name for a function which runs on the GPU.

In addition to executing on the GPU kernels also differ from regular functions in a few ways:

- When you call a kernel it will be called many times in parallel threads. Each thread will have a unique index so you can have each one do something slightly different. Often you pass an array to a kernel and each thread will read one or more items from the array based on it's thread index.
- Kernels cannot return values. Instead it is common to also pass the kernel an output array and have the kernel place its return value into the array at its corresponding thread index.
- The number of parallel threads is configurable via the thread hierarchy (threads, blocks and grids). Ultimately this is limited by your hardware.

#### What is a thread hierarchy (threads, blocks and grids)?

When you call a CUDA kernel it runs n times in n threads.

Threads are grouped into blocks, and the maximum number of threads per block is 1024. 

You can have any number of blocks and these are grouped together into a grid.

All threads in a block have accessed to some shared memory, and can synchronise which means they can wait for them all to reach a specific point in the function before continuing.

When we call our kernel we need to decide how many times it should be run. To do this we pass the thread and block sizes. So if we have an array with 1m items which need to be processed we could make our thread size 1000 and our block size 1000. The number of blocks that a GPU can process at any one time varies by model, but all blocks will be queued when you call your kernel.

Check out [this post](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) for more information.