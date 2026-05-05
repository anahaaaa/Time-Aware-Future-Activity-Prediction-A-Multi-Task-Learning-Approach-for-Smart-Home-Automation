# AI System for Predictive Human Activity Modeling

Human Activity Recognition (HAR) in smart environments has traditionally focused on identifying the current activity of a user based on sensor data. However, many real-world applications, such as assisted living, healthcare monitoring, and smart automation—require systems that go beyond recognition and enable anticipation of future actions.

In a smart home setup, ambient sensors continuously generate event streams reflecting user interactions with the environment. While it is feasible to infer the current activity from these signals, it remains significantly more challenging to:

  - Predict the next activity a user is likely to perform
  -  Estimate when that activity will begin

These two tasks - activity forecasting and time-to-event prediction, are critical for building proactive systems.

## Core Challenges
Several challenges make this problem non-trivial:

1. Temporal Dependency

Human activities are inherently sequential. The next activity depends not only on the current state but also on historical context over time.

2. Sensor Interaction Complexity

Smart home environments involve multiple sensors whose interactions form complex patterns. These relationships are better modeled as a graph structure rather than independent signals.

3. Uncertain Time Intervals

The time gap between activities is highly variable and often skewed, making continuous time prediction unstable without proper modeling.

4. Class Imbalance and Noise

Certain activities occur far more frequently than others, and sensor data may include noise or missing values, affecting model robustness.

## Objectives 

This project aims to design a system that:

- Predicts the next human activity from a sequence of sensor events
- Estimates the time until the next activity begins
- Effectively captures both:
    1. Spatial relationships between sensors
    2. Temporal dependencies across event sequences
 
## Approach Overview

To address these challenges, the problem is formulated as a multi-task learning problem combining:

  1. Activity classification (what happens next)
  2. Time prediction (when it happens)

The system leverages:

- Graph-based modeling to capture sensor relationships
- Sequence modeling to learn temporal patterns
- Discrete time binning to stabilize time prediction

## Pipeline
The system is designed as a fully modular end-to-end pipeline:

1. Data Processing
  - Load raw sensor logs (CSV / TXT)
  - Clean and normalize sensor values
  - Encode categorical variables (sensor IDs, activities)
  - Generate temporal features:
      * time gaps (delta_t)
      * cyclic encoding (hour/day sin-cos)
      * previous activity context
2. Graph Construction
  - Build a sensor interaction graph using transition probabilities
  - Nodes = sensors
  - Edges = co-occurrence / transition strength
  - Graph captures spatial relationships between sensors
3. Sequence Generation
  - Convert event stream → sliding window sequences
  - Each window → graph representation
  - Target:
      * next activity label
      * time until next activity
4. Time Modeling
  - Continuous time → quantile-based bins
  - Solves skewed distribution problem
  - Enables stable classification-based prediction
5. Dataset + Dataloader
  - Custom SequenceDataset
  - Custom collate_fn for graph sequences
6. Training Pipeline
  - Two-stage training:
  - Activity prediction
  - Time prediction (conditioned on activity)
  - Class imbalance handling
  - Learning rate scheduling + early stopping
## Model Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Model Architecture" width="700"/>
  <br/>
  <em>Figure: Hybrid GNN + BiLSTM architecture for activity and time prediction</em>
</p>
