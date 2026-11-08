"""Task 2.7 toy part - Hyperband on Fashion-MNIST, searching learning rate and units."""
import keras_tuner as kt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
(x, y), (xt, yt) = tf.keras.datasets.fashion_mnist.load_data()
x, xt = x / 255.0, xt / 255.0


def build(hp):
    m = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=6, factor=3, seed=42,
                     directory="tuner_toy", project_name="fmnist", overwrite=True)
tuner.search(x, y, validation_split=0.2, verbose=0)
tuner.results_summary(num_trials=5)
best = tuner.get_best_hyperparameters()[0]
print(f"\nBEST: units={best.get('units')}  lr={best.get('lr')}")
model = tuner.hypermodel.build(best)
model.fit(x, y, epochs=5, validation_split=0.2, verbose=0)
print("test accuracy with best settings:", round(model.evaluate(xt, yt, verbose=0)[1], 4))
