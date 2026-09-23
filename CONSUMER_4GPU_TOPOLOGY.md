# Consumer-board 4-GPU topology: PCIe 5.0 x8 / x8 / x4 / x4

This project was not tested on a workstation or server platform with four conventional full-length GPU slots.

The host is an AM5 consumer desktop built around a Gigabyte X870E AORUS MASTER X3D ICE. To explore a fourth GPU, two SSD-capable M.2/NVMe paths were repurposed with NVMe-to-PCIe adapters instead of being reserved for storage.

The resulting target topology is:

```text
RTX 5070 Ti 16 GB   PCIe 5.0 x8
RTX 5070 Ti 16 GB   PCIe 5.0 x8
RTX 5070 Ti 16 GB   PCIe 5.0 x4
RTX 5060 Ti 16 GB   PCIe 5.0 x4
```

In other words: this is a consumer-board inference machine that deliberately trades some SSD expansion flexibility for four GPU links.

## Why test something this awkward?

Aggregate VRAM alone does not tell you whether another GPU is useful for LLM inference. On a consumer platform, the answer also depends on link width, PCIe topology, peer-access behavior, stage balance, model placement and the amount of traffic introduced by the extra device.

That makes the fourth RTX 5060 Ti an unusually useful experiment. It is both a capacity increase and an intentionally asymmetric stage: slower than the three RTX 5070 Ti cards and attached through a x4 link.

This gives the setup a practical question to answer:

> When does adding another 16 GB consumer GPU help more than the extra stage and PCIe traffic hurt?

## What happened in the model-stage test

On the tested PHB/no-P2P topology, adding the RTX 5060 Ti as a fourth model stage increased residency but **reduced end-to-end throughput**.

That negative result is why the promoted Qwen3.8-Flash-Next serving configuration currently uses only the three RTX 5070 Ti devices.

```text
promoted serving path:
3 × RTX 5070 Ti

not promoted:
3 × RTX 5070 Ti + RTX 5060 Ti as a fourth model stage
```

The important lesson is not that a fourth GPU is universally bad. The result is specific to this hardware, topology, model and runtime configuration. It does show that on consumer multi-GPU systems, **more resident weights do not automatically mean more throughput**.

## Why the RTX 5060 Ti may still be useful

The failed fourth-stage test does not make the card useless. It makes a different placement strategy more interesting.

Possible auxiliary roles include:

- a dedicated draft/MTP device;
- a separate small-model server;
- embeddings, reranking, ASR/TTS or other side workloads;
- experiments that avoid placing the RTX 5060 Ti directly on the main model's critical decode path.

Those auxiliary-role comparisons have **not yet been validated as controlled A/B tests**, so this repository does not claim that any one of them is faster yet.

## Why this topology is worth documenting

Most people building a consumer desktop will not buy multiple NVMe-to-PCIe adapters, give up convenient SSD slots, and wire four GPUs into an AM5 board just to discover where multi-GPU inference stops scaling.

That is exactly why the result can be useful to others.

A reader considering a similar build can use these experiments to answer questions before buying hardware:

- Is x4 enough for the role I want to assign to a GPU?
- Does the fourth card belong on the main model path or outside it?
- Is extra VRAM worth another pipeline stage?
- How much does PHB/no-P2P topology matter in practice?
- Is sacrificing M.2 capacity for another GPU link actually worthwhile?

The goal is not to present this machine as a universal reference design. It is to provide measured data from a topology that is uncommon enough that spec sheets alone are not very helpful.

## Current takeaway

For the present Qwen3.8-Flash-Next recipe:

```text
3 × RTX 5070 Ti on the main serving path  -> promoted
RTX 5060 Ti as a fourth model stage       -> tested, slower, not promoted
RTX 5060 Ti as an auxiliary device        -> promising experiment, not yet proven
```

That distinction is intentional. This repository records both the configuration that won and the expensive-looking ideas that did not.
