import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

## Load integrated cohort data
data = pd.read_csv("Cohort_A_Integrated.csv")

## Select features for VAE
exclude_cols = ["ID", "Cohort", "AD_Conversion", "Time_to_Event", "Followup_Years"]
feature_cols = [col for col in data.columns if col not in exclude_cols]
features = data[feature_cols].select_dtypes(include=[np.number]).copy()
features = features.dropna(axis=1, how="all").fillna(features.mean())

print(f"Input features: {features.shape[1]}")
print(f"Sample size: {features.shape[0]}")

## Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

## VAE architecture parameters
input_dim = X_scaled.shape[1]
latent_dim = 10
intermediate_dim = 64

## Encoder
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(intermediate_dim, activation="relu")(encoder_inputs)
x = layers.Dense(32, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

## Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

## Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32, activation="relu")(latent_inputs)
x = layers.Dense(intermediate_dim, activation="relu")(x)
decoder_outputs = layers.Dense(input_dim, activation="linear")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

## VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

## Build and compile VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

## Train VAE
print("\nTraining VAE...")
history = vae.fit(
    X_scaled,
    epochs=100,
    batch_size=32,
    verbose=0
)

print(f"Final loss: {history.history['loss'][-1]:.4f}")

## Extract latent embeddings
z_mean, z_log_var, z_sample = encoder.predict(X_scaled)

## K-means clustering on latent space
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
cluster_labels = kmeans.fit_predict(z_mean)

print(f"\nCluster distribution:")
for i in range(n_clusters):
    count = np.sum(cluster_labels == i)
    print(f"  Cluster {i}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

## Save results
latent_df = pd.DataFrame(z_mean, columns=[f"Latent_{i+1}" for i in range(latent_dim)])
latent_df.insert(0, "ID", data["ID"].values)
latent_df["Cluster_Labels"] = cluster_labels

latent_df.to_csv("VAE_latent_embeddings.csv", index=False)

## Save cluster results
cluster_df = pd.DataFrame({
    "ID": data["ID"].values,
    "Cluster_Labels": cluster_labels
})
cluster_df.to_csv("cluster_results.csv", index=False)

print(f"\nOutputs saved:")
print("  - VAE_latent_embeddings.csv")
print("  - cluster_results.csv")
