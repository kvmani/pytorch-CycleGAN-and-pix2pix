# MicroI2I Documentation

MicroI2I is a scientific microscopy image-to-image translation platform built from the pix2pix and CycleGAN lineage.
It is intended to support research workflows and teach the principles behind modern generative image translation.

```{image} diagrams/code_architecture.svg
:alt: MicroI2I module architecture
:class: architecture-diagram
```

## Start Here

::::{grid} 2
:::{grid-item-card} Run The Code
:link: usage_commands
:link-type: doc
Use canonical commands for dataset preparation, training, inference, evaluation, and registry validation.
:::
:::{grid-item-card} Learn The Principles
:link: teaching_gan_principles
:link-type: doc
Understand GANs, paired/unpaired translation, objectives, and practical training terms.
:::
:::{grid-item-card} Review The Architecture
:link: architecture
:link-type: doc
See how CLI, configs, manifests, dataops, model backends, and evaluation modules connect.
:::
:::{grid-item-card} Scientific Validation
:link: scientific_validation
:link-type: doc
Study metrics, assumptions, and how outputs should be interpreted for microscopy research.
:::
::::

```{toctree}
:maxdepth: 2
:caption: Foundations

mission_statement
documentation_principles
code_provenance_and_manifests
quality_gates
developer_guide
repository_blueprint
current_state_audit
development_roadmap
```

```{toctree}
:maxdepth: 2
:caption: Teaching And Algorithms

teaching_gan_principles
gan_formulations
model_architectures
metrics_formulations
learning_terms
glossary
```

```{toctree}
:maxdepth: 2
:caption: Workflows

architecture
usage_commands
training_data_requirements
model_backend_interface
model_registry
scientific_validation
tutorials/01_prepare_paired_microscopy_dataset
tutorials/02_train_and_infer_pix2pix
tutorials/03_train_and_infer_cyclegan
tutorials/04_dataset_qa_before_training
tutorials/05_inference_batch_review
tutorials/06_smoke_workflows
tutorials/07_ebsd_kikuchi_domain_wrappers
```

```{toctree}
:maxdepth: 1
:caption: Legacy Reference

overview
datasets
tips
qa
windows_cyclegan_training
```
