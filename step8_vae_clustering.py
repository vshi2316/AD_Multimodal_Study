import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("VAE Clustering for ADNI Cohort")
print("="*70)

## Load integrated cohort data
data = pd.read_csv("Cohort_A_Integrated.csv")
print(f"\nTotal samples: {len(data)}")

## Select features for VAE (exclude ID and outcome variables)
exclude_cols = ["ID", "Cohort", "AD_Conversion", "Time_to_Event", "Followup_Years"]
feature_cols = [col for col in data.columns if col not in exclude_cols]
features = data[feature_cols].select_dtypes(include=[np.number]).copy()

## Handle missing values
features = features.dropna(axis=1, how="all")
for col in features.columns:
    features[col] = features[col].fillna(features[col].median())

print(f"Input features: {features.shape[1]}")
print(f"Sample size: {features.shape[0]}")

## Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

## Train/validation split
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

## VAE architecture parameters
input_dim = X_scaled.shape[1]
latent_dim = 10
hidden_dim1 = 128
hidden_dim2 = 64
batch_size = 32

print(f"\nVAE Architecture: {input_dim} → {hidden_dim1} → {hidden_dim2} → {latent_dim}")

## Encoder
encoder_input = layers.Input(shape=(input_dim,))
h = layers.Dense(hidden_dim1, activation='relu')(encoder_input)
h = layers.Dense(hidden_dim2, activation='relu')(h)
z_mean = layers.Dense(latent_dim, name='z_mean')(h)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)

## Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

## Decoder
decoder_input = layers.Input(shape=(latent_dim,))
h = layers.Dense(hidden_dim2, activation='relu')(decoder_input)
h = layers.Dense(hidden_dim1, activation='relu')(h)
decoder_output = layers.Dense(input_dim, activation='linear')(h)
decoder = Model(decoder_input, decoder_output, name='decoder')

## VAE model
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
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
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
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

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

## Train VAE
print("\nTraining VAE...")
history = vae.fit(
    X_train,
    epochs=150,
    batch_size=batch_size,
    validation_data=(X_val, None),
    verbose=0
)

print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

## Extract latent embeddings
z_mean_encoded, _, _ = encoder.predict(X_scaled, verbose=0)
print(f"\nLatent space shape: {z_mean_encoded.shape}")

## K-means clustering - evaluate K from 2 to 5
print("\nEvaluating optimal number of clusters...")
silhouette_scores = []
k_range = range(2, 6)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
    labels = kmeans.fit_predict(z_mean_encoded)
    score = silhouette_score(z_mean_encoded, labels)
    silhouette_scores.append(score)
    print(f"  K={k}: Silhouette={score:.3f}")

## Select optimal K
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K selected: {optimal_k}")

## Final clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=100)
cluster_labels = kmeans_final.fit_predict(z_mean_encoded)

print(f"\nCluster distribution:")
for i in range(optimal_k):
    count = np.sum(cluster_labels == i)
    print(f"  Cluster {i}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

## Save results
# Latent embeddings with cluster labels
latent_df = pd.DataFrame(
    z_mean_encoded, 
    columns=[f'Latent_{i+1}' for i in range(latent_dim)]
)
latent_df.insert(0, 'ID', data['ID'].values)
latent_df['Cluster_Labels'] = cluster_labels

if 'AD_Conversion' in data.columns:
    latent_df['AD_Conversion'] = data['AD_Conversion'].values

latent_df.to_csv("latent_encoded.csv", index=False)

# Cluster results
cluster_df = pd.DataFrame({
    'ID': data['ID'].values,
    'Cluster_Labels': cluster_labels
})

if 'AD_Conversion' in data.columns:
    cluster_df['AD_Conversion'] = data['AD_Conversion'].values

cluster_df.to_csv("cluster_results.csv", index=False)

print("\n" + "="*70)
print("VAE clustering complete!")
print("="*70)
print("\nOutputs saved:")
print("  - latent_encoded.csv")
print("  - cluster_results.csv")
