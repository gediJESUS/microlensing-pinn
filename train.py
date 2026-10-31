"""
Physics-Informed Neural Network (PINN) Training Pipeline
=========================================================
Research Thesis: "Integration of Deep Learning Models for Detection and 
                  Characterization of Exoplanets via Gravitational Microlensing"
Author: Gedion Melese Ashebo
Institution: Adama Science and Technology University (ASTU)
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------
# Auto-Install MulensModel if missing (Handles Colab / Clean Environments)
# ------------------------------------------------------------------------------
try:
    import MulensModel as mm
except ModuleNotFoundError:
    print(" Installing specialized astrophysics dependency: MulensModel...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "MulensModel", "--quiet"])
    import MulensModel as mm

# Set TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==============================================================================
# 1. DIFFERENTIABLE PHYSICS ENGINE
# ==============================================================================
@tf.function
def tf_paczynski_engine(u0, tE, t0=50.0):
    """
    Analytic Point-Source Point-Lens (PSPL) Engine.
    Computes theoretical magnification directly on GPU for backpropagation.
    """
    t_grid = tf.linspace(0.0, 100.0, 1024)
    u = tf.sqrt(
        tf.square(u0[:, tf.newaxis]) +
        tf.square((t_grid - t0) / (tE[:, tf.newaxis] + 1e-8))
    )
    mag = (tf.square(u) + 2.0) / (u * tf.sqrt(tf.square(u) + 4.0) + 1e-8)
    return (mag - 1.0) / 0.5  # Standardized to input manifold


# ==============================================================================
# 2. SYNTHETIC DATA GENERATOR (MulensModel Simulation)
# ==============================================================================
def data_generator(batch_size=32, planet_ratio=0.5):
    """
    Dynamically generates synthetic microlensing events using MulensModel.
    Samples from the parameter manifold:
        q in [10^-4.5, 10^-2.5], s in [0.7, 1.3], u0 in [0.01, 0.4], tE in [10, 50]
    """
    while True:
        X, Yd, Yr = [], [], []
        for _ in range(batch_size):
            is_planet = np.random.rand() < planet_ratio
            u0 = np.random.uniform(0.01, 0.4)
            tE = np.random.uniform(10.0, 50.0)
            t0 = 50.0

            if is_planet:
                q = 10 ** np.random.uniform(-4.5, -2.5)
                s = np.random.uniform(0.7, 1.3)
                alpha = np.random.uniform(0, 360)
                try:
                    m = mm.Model({'t_0': t0, 'u_0': u0, 't_E': tE, 'q': q, 's': s, 'alpha': alpha})
                    m.set_magnification_methods([0., 'point_source', 100.])
                    mag = m.get_magnification(np.linspace(0, 100, 1024))
                    
                    # Normalized labels: [u0, tE, log(q), s, alpha]
                    Yr.append([(u0 - 0.01) / 0.39, (tE - 10) / 40.0, (np.log10(q) + 4.5) / 2.0, (s - 0.7) / 0.6, alpha / 360.0])
                    Yd.append(1.0)
                except Exception:
                    continue
            else:
                u_dist = np.sqrt(u0**2 + ((np.linspace(0, 100, 1024) - t0) / tE)**2)
                mag = (u_dist**2 + 2) / (u_dist * np.sqrt(u_dist**2 + 4))
                Yr.append([(u0 - 0.01) / 0.39, (tE - 10) / 40.0, 0.5, 0.5, 0.5])
                Yd.append(0.0)

            # Realistic photometric noise injection
            noise = np.random.normal(0, 0.015, 1024)
            norm_flux = (mag + noise - 1.0) / 0.5
            X.append(norm_flux[..., None])

        yield np.array(X, dtype=np.float32), np.array(Yd, dtype=np.float32), np.array(Yr, dtype=np.float32)


# ==============================================================================
# 3. 1D-RESNET ARCHITECTURE
# ==============================================================================
def res_block(x_in, filters):
    """Residual convolutional block with identity shortcut connection."""
    shortcut = x_in
    x = layers.Conv1D(filters, 7, padding="same", activation="relu")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, 7, padding="same")(x)
    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)

def build_pinn_architecture():
    """Builds the 1D-ResNet Dual-Head Network."""
    inp = layers.Input(shape=(1024, 1), name="LightCurve_Input")
    
    # Feature extraction backbone
    x = layers.Conv1D(64, 11, padding="same", activation="relu")(inp)
    x = res_block(x, 64)
    x = layers.MaxPooling1D(4)(x)
    
    x = layers.Conv1D(128, 7, padding="same", activation="relu")(x)
    x = res_block(x, 128)
    x = layers.GlobalAveragePooling1D()(x)
    features = layers.Dense(256, activation='relu')(x)

    # Dual heads: Binary Detection & Physical Parameter Regression
    det_out = layers.Dense(1, activation='sigmoid', name='Detection')(features)
    reg_out = layers.Dense(5, activation='sigmoid', name='Physics_Params')(features)

    return models.Model(inputs=inp, outputs=[det_out, reg_out], name="PINN_Microlensing")


# ==============================================================================
# 4. CUSTOM PHYSICS-INFORMED TRAINING WRAPPER (TRIPLE-LOSS)
# ==============================================================================
class PINNModel(models.Model):
    def __init__(self, base_model):
        super(PINNModel, self).__init__()
        self.base_model = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def train_step(self, data):
        x, (yd, yr) = data
        with tf.GradientTape() as tape:
            pd, pr = self.base_model(x, training=True)

            # 1. Anomaly Classification Loss (Binary Cross-Entropy)
            loss_det = tf.reduce_mean(tf.keras.losses.binary_crossentropy(yd, pd))

            # 2. Parameter Regression Loss (MAE)
            loss_reg = tf.reduce_mean(tf.abs(yr - pr))

            # 3. Differentiable Physics Consistency Loss (PSPL Reconstruction)
            u0_pred = pr[:, 0] * 0.39 + 0.01
            tE_pred = pr[:, 1] * 40.0 + 10.0
            x_phys = tf_paczynski_engine(u0_pred, tE_pred)
            loss_phys = tf.reduce_mean(tf.square(x[:, :, 0] - x_phys))

            # Multi-objective Triple-Loss
            total_loss = loss_det + (15.0 * loss_reg) + (5.0 * loss_phys)

        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result(), "det_loss": loss_det, "phys_loss": loss_phys}

    def call(self, x):
        return self.base_model(x)


# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train PINN for Gravitational Microlensing")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save weights & logs")
    
    # parse_known_args allows running inside Jupyter / Colab notebooks without conflict
    args, _ = parser.parse_known_args()

    # Create directories
    models_dir = os.path.join(args.output_dir, "models")
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("🛠 Initializing PINN Architecture...")
    core_model = build_pinn_architecture()
    pinn_trainer = PINNModel(core_model)
    pinn_trainer.compile(optimizer=optimizers.Adam(learning_rate=1e-4))

    # Pre-generate validation set
    print("📊 Creating static validation vault (N=320)...")
    val_gen = data_generator(batch_size=320)
    vx, vyd, vyr = next(val_gen)

    # Training marathon
    print(f"🚀 Launching {args.epochs}-epoch training cycle...")
    train_gen = data_generator(batch_size=args.batch_size)
    history = []

    for epoch in range(args.epochs):
        start_time = time.time()
        epoch_losses = []
        
        for _ in range(50):  # 50 gradient steps per epoch
            bx, byd, byr = next(train_gen)
            logs = pinn_trainer.train_step((bx, (byd[:, None], byr)))
            epoch_losses.append(logs['loss'].numpy())

        # Validation evaluation
        vpd, vpr = pinn_trainer.predict(vx, verbose=0)
        auc = roc_auc_score(vyd, vpd.flatten())
        q_mae = np.mean(np.abs(vyr[:, 2] - vpr[:, 2]))
        mean_loss = float(np.mean(epoch_losses))

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {mean_loss:.4f} | AUC: {auc:.4f} | q-MAE: {q_mae:.4f} | Elapsed: {elapsed:.1f}s")

        history.append({"epoch": epoch + 1, "loss": mean_loss, "auc": auc, "q_mae": q_mae})

        # Save history checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            pd.DataFrame(history).to_csv(os.path.join(logs_dir, "training_history.csv"), index=False)

    # Save final model
    save_path = os.path.join(models_dir, "pinn_microlensing_final.h5")
    core_model.save(save_path)
    print(f"✅ Training complete. Final model saved to: {save_path}")


if __name__ == "__main__":
    main()
