"""
Go move prediction — training script.

Trains a dual-head (policy + value) residual network on Go positions,
under a strict budget of fewer than 100,000 trainable parameters.

The network follows an AlphaZero-style design:
    - a shared residual trunk built from depthwise-separable convolutions
      with Squeeze-and-Excitation blocks;
    - a policy head predicting a distribution over the 361 board points;
    - a value head predicting the white player's win probability.

Data is provided by `golois`, a compiled C module used in the course. It
exposes `getBatch` (load a batch of positions from `games.data`) and
`getValidation` (load a fixed validation set). `golois` is distributed with
the course materials and is *not* available on PyPI.

Authors: Tomil Shi, Émile Descroix, Alexis Michel
Deep Learning course — Université Paris Dauphine – PSL (2025–2026)
"""

import gc

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers

import golois  # course-provided C module (data loader), not on PyPI

# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
planes = 31     # input feature planes per board position
moves = 361     # number of board points (19 x 19)
N = 10000       # positions per epoch
batch = 256
filters = 64
epochs = 3000

L2 = 1e-4       # L2 regularization applied to every layer

# --------------------------------------------------------------------------- #
# Data buffers (filled in-place by golois)
# --------------------------------------------------------------------------- #
input_data = np.random.randint(2, size=(N, 19, 19, planes)).astype('float32')
policy = keras.utils.to_categorical(
    np.random.randint(moves, size=(N,))).astype('float32')
value = np.random.randint(2, size=(N,)).astype('float32')
end = np.random.randint(2, size=(N, 19, 19, 2)).astype('float32')
groups = np.zeros((N, 19, 19, 1)).astype('float32')

print("Tensorflow version", tf.__version__)
print("getValidation", flush=True)
golois.getValidation(input_data, policy, value, end)


# --------------------------------------------------------------------------- #
# Data augmentation
# --------------------------------------------------------------------------- #
def augment_batch(x, p, v):
    """Apply a random element of the dihedral group D4 (8 symmetries).

    Go is invariant under 90-degree rotations and horizontal flips, so the
    board and the policy target are transformed together. The value target is
    a global scalar and is left untouched.
    """
    k = np.random.randint(0, 4)     # rotation: 0, 90, 180 or 270 degrees
    flip = np.random.randint(0, 2)  # horizontal flip

    x_aug = np.rot90(x, k, axes=(1, 2))
    p_aug = np.rot90(p.reshape(-1, 19, 19), k, axes=(1, 2)).reshape(-1, 361)

    if flip:
        x_aug = np.flip(x_aug, axis=2).copy()
        p_aug = np.flip(p_aug.reshape(-1, 19, 19), axis=2).reshape(-1, 361).copy()

    return x_aug, p_aug, v


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def se_block(x, filters, ratio):
    """Squeeze-and-Excitation block: recalibrate channels by global content."""
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu',
                      kernel_regularizer=regularizers.l2(L2))(se)
    se = layers.Dense(filters, activation='sigmoid',
                      kernel_regularizer=regularizers.l2(L2))(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def separable_residual_block(x, filters):
    """Residual block of two separable convolutions followed by an SE block."""
    shortcut = x

    x = layers.SeparableConv2D(
        filters, 3, padding='same', use_bias=False,
        depthwise_regularizer=regularizers.l2(L2),
        pointwise_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(
        filters, 3, padding='same', use_bias=False,
        depthwise_regularizer=regularizers.l2(L2),
        pointwise_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)

    x = se_block(x, filters, 8)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
inputs = keras.Input(shape=(19, 19, planes), name='board')

# Entry block: a standard Conv2D mixes the correlated input planes better
# than a separable convolution would.
x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                  kernel_regularizer=regularizers.l2(L2))(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

# Shared residual trunk.
for _ in range(7):
    x = separable_residual_block(x, filters)

# Policy head: distribution over the 361 board points.
policy_head = layers.Conv2D(4, 1, padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(L2))(x)
policy_head = layers.BatchNormalization()(policy_head)
policy_head = layers.Activation('relu')(policy_head)
policy_head = layers.Conv2D(1, 1, padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(L2))(policy_head)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)

# Value head: white player's win probability in [0, 1].
value_head = layers.Conv2D(32, 1, padding='same', use_bias=False,
                           kernel_regularizer=regularizers.l2(L2))(x)
value_head = layers.BatchNormalization()(value_head)
value_head = layers.Activation('relu')(value_head)
value_head = layers.GlobalAveragePooling2D()(value_head)
value_head = layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(L2))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value',
                          kernel_regularizer=regularizers.l2(L2))(value_head)

model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
model.summary()

# --------------------------------------------------------------------------- #
# Optimizer and learning-rate schedule
# --------------------------------------------------------------------------- #
total_steps = epochs * (N // batch)

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=total_steps,
    alpha=1e-6,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss={
        'policy': 'categorical_crossentropy',
        'value': 'binary_crossentropy',
    },
    # Policy accuracy is the main evaluation metric, so it is weighted higher.
    loss_weights={'policy': 2.0, 'value': 1.0},
    metrics={
        'policy': 'categorical_accuracy',
        'value': 'mae',
    },
)

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
for i in range(1, epochs + 1):
    print(f'\nEpoch {i}/{epochs}')

    # Load a fresh batch of positions from games.data.
    golois.getBatch(input_data, policy, value, end, groups, i * N)

    # Apply a random symmetry from D4.
    x_aug, p_aug, v_aug = augment_batch(input_data, policy, value)

    model.fit(
        x_aug,
        [p_aug, v_aug],
        epochs=1,
        batch_size=batch,
        verbose=1,
    )

    if i % 10 == 0:
        gc.collect()

    # Every 50 epochs: full validation + checkpoint.
    if i % 50 == 0:
        golois.getValidation(input_data, policy, value, end)
        val = model.evaluate(input_data, [policy, value],
                             verbose=0, batch_size=batch)
        print(f"Validation epoch {i} : loss={val[0]:.4f} | "
              f"policy_acc={val[3]:.4f} | value_mae={val[4]:.4f}")
        model.save(f'test_{i}.h5')

# Final model. For the course submission this file is renamed to
# Emile_DESCROIX-Alexis_MICHEL-Tomil_SHI.h5
model.save('test.h5')
