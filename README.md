# Cross-Controller Transfer via In-Context Behavior Adaptation
[Project Website](https://controller-transfer.github.io)
This repository accompanies **Cross-Controller Transfer via In-Context Behavior Adaptation**, a research project on learning robot policies that transfer across low-level controllers with different dynamics and interaction behaviors.

## Overview

Robot policies are typically trained under a particular low-level controller. However, the same high-level action command can produce substantially different robot motions depending on controller type, gains, damping, compliance, and hardware. Naively combining demonstrations from these settings creates inconsistent action-to-outcome relationships and can degrade policy learning.

We study **controller transfer**: can a single visuomotor policy use data collected under different controllers and adapt to a new controller without retraining a separate task policy?

Our approach conditions a vision-language-action (VLA) policy on a compact representation of the controller's behavior, inferred **in context** from examples of commanded actions and observed robot responses. This enables the policy to account for controller-dependent execution and adapt its task strategy, particularly during contact-rich manipulation.

## Key Idea

The controller is part of the system that determines how a policy's commands are realized. Instead of treating controller variation as unstructured noise, we provide the policy with evidence about the action-to-motion mapping it is operating under.

1. **Collect controller context.** Record short interaction histories containing commanded actions and resulting robot motion.
2. **Encode controller behavior.** A Perceiver-style context encoder compresses variable-length interaction histories into fixed-length context tokens. Controller metadata, including natural-language descriptions when available, can provide additional conditioning.
3. **Condition the task policy.** A shared VLA policy uses visual observations, task instructions, and controller context to select actions appropriate for the current controller.
4. **Transfer to new settings.** At inference time, controller-characterization examples allow adaptation to new controller parameters or types without requiring a separately trained policy for each controller.

The objective is not only to compensate for action scaling or tracking error. Controller properties can change *which behavior succeeds*: for example, compliant contact may support a search strategy during insertion, while a stiffer controller may require a different approach to avoid jamming.

## Research Questions

- Can demonstrations collected under heterogeneous controllers be pooled productively?
- How should a policy represent controller-dependent behavior rather than merely identify controller parameters?
- Does in-context controller characterization generalize to unseen gains, damping, and controller types?
- Can controller awareness improve strategy selection for contact-rich tasks?
- How much characterization data is needed to adapt to a new controller?

## Method

### Controller characterization

We form a context sequence from pairs of commands and their observed consequences. These examples characterize how the controlled robot responds under the current low-level controller. A learned context encoder converts the history into conditioning tokens that can be consumed by the task policy.

### Controller-conditioned policy

The task policy predicts robot actions from its usual observations and language instruction while attending to controller context. Training across controller variations encourages the policy to separate task intent from the controller-specific means of execution.

### In-context adaptation

For a new controller, the model receives characterization interactions from that controller. Adaptation takes place through the conditioning context rather than task-specific gradient updates.

## Evaluation

The project studies transfer across **six controller types**, including variations in gain and compliance, on **five real-world manipulation tasks**. Evaluations include continuous controller-parameter variation and contact-rich tasks such as **USB insertion** and **surface wiping**.

The central comparisons include:

| Approach | Controller information | Main question |
| --- | --- | --- |
| Naive joint training | None beyond pooled demonstrations | Can a policy absorb controller variation implicitly? |
| Ground-truth controller conditioning | Explicit controller parameters | How useful is privileged controller knowledge? |
| System identification | Estimated controller/dynamics parameters | Is identifying dynamics sufficient for behavioral transfer? |
| Synthetic relabeling | Adjusted actions or data | Can data harmonization resolve controller mismatch? |
| In-context behavior adaptation (ours) | Command-response context, with optional metadata | Can observed controller behavior support generalizable policy adaptation? |

Evaluation focuses on task success, generalization to held-out controller settings, contact-rich behavior, and adaptation as the amount of controller context varies.

## Main Takeaway

**Controller choice changes more than the mapping between a commanded action and robot motion—it can change the task behavior that is effective.** A policy that conditions on observed controller behavior can use heterogeneous demonstrations more effectively and adapt its execution strategy across controllers.

## Paper

*Cross-Controller Transfer via In-Context Behavior Adaptation* (ICRA 2027 submission).

<!-- Add paper, project website, and citation links when publicly available. -->

## Code and Reproducibility

This README describes the research project at a high level. Installation, dataset preparation, training, evaluation, checkpoint downloads, and hardware-specific instructions should be added alongside the corresponding released code and configurations.

