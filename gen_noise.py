import numpy as np
import pandas as pd

# Constants
ROWS = 1000
COLS = 10

def generate_dataset(rows, cols):
    """Generate synthetic dataset for modeling."""
    X = 2 * np.random.rand(rows, cols).astype(np.float16)
    w = 10 * np.random.rand(cols).astype(np.float16)
    y_actual = np.dot(X, w).astype(np.float32)
    y_actual = np.reshape(y_actual, (rows, 1)).copy()
    return X, w, y_actual

def add_noise(y_actual, rows):
    """Add different types of noise to the dataset."""
    s0, s1, s2, s3 = int(0.3 * rows), int(0.2 * rows), int(0.2 * rows), int(0.3 * rows)
    noises = [
        (50.0 * np.random.normal(3, 1, (s0, 1))).astype(np.float32),
        (25.0 * np.random.normal(-2, 1, (s1, 1))).astype(np.float32),
        (2.0 * np.random.normal(-1, 1, (s2, 1))).astype(np.float32),
        (0.1 * np.random.normal(0, 1, (s3, 1))).astype(np.float32),
    ]
    noise = np.vstack(noises)
    np.random.shuffle(noise)
    return y_actual + noise

def save_dataset(X, y, prefix='df'):
    """Save dataset to CSV."""
    pd.DataFrame(X).to_csv(f'{prefix}_X.csv', index=False, header=None)
    pd.DataFrame(y).to_csv(f'{prefix}_y.csv', index=False, header=None)

def linear_regression(X, y, cols):
    """Fit and evaluate linear regression model."""
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    coefficients = np.reshape(model.coef_, (cols, 1)).copy()
    return coefficients

# Generate synthetic dataset
X, w, y_actual = generate_dataset(ROWS, COLS)
y = add_noise(y_actual, ROWS)

# Save generated datasets
save_dataset(X, y, 'train')
save_dataset(X, y_actual, 'test')
