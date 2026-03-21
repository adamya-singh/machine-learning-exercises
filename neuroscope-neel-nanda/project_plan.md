# OLMoE Neuroscope Mini-Project Map

## Project Goal

This project will adapt the existing neuroscope-style idea from a dense GPT-2 neuron viewer into an OLMoE expert-routing analysis project.

The main goal is to:

- use `allenai/OLMoE-1B-7B-0924` instead of `gpt2-small`
- examine expert-level routing behavior instead of dense neuron activations
- use the DBPedia14 dataset as a probe dataset
- investigate whether different DBPedia topics show distinct expert and layer routing patterns

This project is intended to study how OLMoE processes different topic categories, not to prove exactly which expert originally learned a topic during pretraining.

## Core Research Question And Interpretation Boundary

The core question is:

Which experts and layers does OLMoE route DBPedia14 topic examples through, and do different topics show distinct or overlapping routing patterns?

The interpretation boundary should stay explicit:

- this project measures which experts and layers are used when processing topic examples
- this project does not prove that a specific expert originally learned a topic during training
- the safest conclusions are about routing specialization and similarity of processing patterns
- if two topics are consistently routed through similar experts, that suggests the model may be processing them similarly

## High-Level Architecture

The project architecture has the following components:

- dataset ingestion: load DBPedia14 samples and topic labels
- preprocessing: convert each example into text suitable for OLMoE inference
- model inference: run OLMoE with `output_router_logits=True`
- signal extraction: capture per-layer router probabilities and top-k expert assignments for each token
- storage: save routing outputs together with metadata such as example ID, topic label, token position, and layer index
- aggregation: summarize token-level and example-level routing into topic-level statistics
- visualization: build topic-by-layer-by-expert views and token-level inspection views inspired by the neuroscope workflow
- analysis: compare expert usage across DBPedia topics and identify recurring routing patterns

## Data Flow Overview

1. Select a subset of DBPedia14 examples grouped by topic label.
2. Format each example into input text for OLMoE.
3. Tokenize the text with the OLMoE tokenizer.
4. Run the model and request router outputs for every MoE layer.
5. For each token and layer, capture:
   - the router probability distribution over experts
   - the top-k selected experts
6. Store these outputs alongside topic and token metadata.
7. Aggregate across tokens and examples to create topic-level summaries such as:
   - average routing probability per expert
   - expert selection frequency by layer
   - dominant experts for each topic
8. Visualize the aggregated summaries and inspect selected examples interactively.
9. Interpret overlaps and differences across topics as evidence of similar or different routing behavior.

## Phased Implementation Direction

### Phase 1: Router-Level Analysis

Build the initial pipeline around router probabilities and top-k expert assignments.

This phase should:

- load DBPedia14 examples
- run OLMoE inference
- extract router outputs per layer and token
- save routing statistics
- aggregate by topic
- produce summary tables and visualizations

This is the easiest, cleanest, and most defensible first version of the project.

### Phase 2: Expert Contribution Analysis

Extend the project beyond routing by estimating how strongly selected experts contribute to the final MoE output.

This phase should be treated as a later extension because it will likely require deeper model hooks or custom instrumentation beyond simply requesting router logits.

## Expected Outputs

The project should eventually produce:

- a reusable script or notebook for running DBPedia14 examples through OLMoE
- saved routing statistics for each example, token, and layer
- topic-level summary tables
- visualizations showing expert usage across topics and layers
- a short conclusions section with cautious claims about routing specialization

## Validation Checklist

When implementation starts, verify that:

- DBPedia14 loads with usable text fields and labels
- OLMoE inference returns router outputs for each MoE layer
- per-token routing data can be saved with topic metadata
- aggregation produces stable topic-level routing patterns on a small sample
- visualizations clearly distinguish topic, layer, and expert dimensions

## Assumptions

- the initial implementation uses router probabilities rather than expert-output magnitudes
- the project is exploratory and interpretive, not a causal proof of what was learned during training
- the existing neuroscope notebook is mainly a visualization inspiration rather than the final technical backbone
- this file should remain high-level and directional rather than turning into a detailed implementation spec

## Rough Starting Idea

I want to use OLMoE instead of gpt 2 small, and I want to examine activations at the expert level instead of the neuron level. I want to use the DBPedia14 dataset. I want to do the mini project I have described below.


project idea:
I found this interactive neuroscope colab notebook from neel nanda (ive given it to you as a .py so you can read it all and understand what Im talking about). Im thinking of taking a small dataset, and running it topic by topic through the llm while looking at all the layers for each topic using the neuroscope and recording the activations for each piece of the data. then, I will be able to see which topics from the dataset were learned by which experts, and with some basic analysis I can reach a conclusion like "this part of the dataset was learned by this expert, this expert learned these two parts of the dataset, etc" and I can hypothesize things like "the model learned these two topics to be similar (since theyre learned by the same expert)". Does this idea make sense? Is it relatively easy to implement building on this colab notebook?
