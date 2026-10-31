# Physics-Informed Deep Learning for Gravitational Microlensing

This repository contains the core training pipeline developed for the M.Sc. thesis:
> **"Integration of Deep Learning Models for Detection and Characterization of Exoplanets via Gravitational Microlensing"**  
> **Author:** Gedion Melese Ashebo  
> **Advisors:** Dr. Seblu Humne Negu, Dr. Endale Gemechu Bedada  
> **Department:** Applied Physics (Astrophysics), Adama Science and Technology University (ASTU)

---

## Project Overview

Traditional exoplanet characterization in microlensing surveys relies on computationally intensive grid-searches and MCMC chains. This project implements a **1D-Residual Convolutional Neural Network (1D-ResNet)** integrated with a differentiable analytical **Point-Source Point-Lens (PSPL)** forward model within TensorFlow.

### Key Contributions:
- **Analytic Physics Loss:** Computes theoretical Paczyński (1986) magnification directly on the GPU to penalize non-physical parameter proposals via backpropagation.
- **Sub-Second Inference:** Enables real-time anomaly screening suitable for upcoming high-cadence streams like the Nancy Grace Roman Space Telescope.
- **Survey Validation:** Audited on archival OGLE-IV and KMTNet light curves, achieving zero false alarms on the non-lensed stellar control cohort.

---

## Repository Structure

- `train.py`: Full training pipeline (synthetic generator, 1D-ResNet, differentiable PSPL engine, and multi-objective loss).
- `requirements.txt`: Python package dependencies.

---

## Quickstart

```bash
# Clone this repository
git clone https://github.com/gediJESUS/microlensing-pinn.git
cd microlensing-pinn

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python train.py --epochs 250 --batch_size 32
