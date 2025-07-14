from datetime import datetime, timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import time
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import warnings
import pickle
warnings.filterwarnings('ignore')

POWER_CONSUMPTION_KW = 0.1
def calculate_implied_volatility(data, window=20):
    """
    Calculate realistic implied volatility using GARCH-inspired approach:
    1. Base realized volatility
    2. GARCH-like conditional variance modeling
    3. One-sigma confidence bounds for upper range estimation
    """
    import numpy as np
    
    # Calculate returns
    returns = data['Close'].pct_change().dropna()
    
    # Base realized volatility (annualized)
    realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    # GARCH-inspired conditional variance modeling
    # Simple GARCH(1,1) approximation: σ²ₜ = ω + α*εₜ₋₁² + β*σ²ₜ₋₁
    
    # Initialize parameters (typical GARCH values)
    omega = 0.00001  # Long-term variance component
    alpha = 0.1      # Short-term shock coefficient  
    beta = 0.85      # Persistence coefficient
    
    # Ensure α + β < 1 for stationarity
    if alpha + beta >= 1:
        alpha = 0.08
        beta = 0.87
    
    # Initialize conditional variance series
    conditional_var = np.zeros(len(returns))
    
    # Set initial variance to sample variance
    initial_var = np.var(returns.dropna())
    conditional_var[0] = initial_var
    
    # Calculate GARCH conditional variance
    for i in range(1, len(returns)):
        if not np.isnan(returns.iloc[i-1]):
            # GARCH equation: σ²ₜ = ω + α*εₜ₋₁² + β*σ²ₜ₋₁
            conditional_var[i] = (omega + 
                                alpha * (returns.iloc[i-1] ** 2) + 
                                beta * conditional_var[i-1])
        else:
            conditional_var[i] = conditional_var[i-1]
    
    # Convert conditional variance to volatility (annualized)
    conditional_vol = np.sqrt(conditional_var) * np.sqrt(252)
    
    # Calculate one-sigma upper bound using rolling statistics
    # This gives us the expected range of fluctuation
    rolling_mean_vol = pd.Series(conditional_vol).rolling(window=window).mean()
    rolling_std_vol = pd.Series(conditional_vol).rolling(window=window).std()
    
    # One-sigma upper bound (68% confidence interval upper limit)
    one_sigma_upper = rolling_mean_vol + rolling_std_vol
    
    # Smooth the volatility estimate using exponential moving average
    # This reduces noise while preserving trend changes
    smoothing_factor = 2.0 / (window + 1)  # EMA smoothing
    smoothed_vol = pd.Series(one_sigma_upper).ewm(alpha=smoothing_factor).mean()
    
    # Apply reasonable bounds
    # Most stocks have IV between 10% and 100%, extreme cases up to 150%
    bounded_vol = smoothed_vol.clip(0.05, 1.5)  # 5% to 150% annualized
    
    # Handle NaN values by forward filling, then backward filling
    final_vol = bounded_vol.fillna(method='ffill').fillna(method='bfill')
    
    # If still NaN (edge case), use median of non-NaN values
    if final_vol.isna().any():
        median_vol = final_vol.median()
        if np.isnan(median_vol):
            median_vol = 0.2  # 20% default volatility
        final_vol = final_vol.fillna(median_vol)
    
    # Ensure output length matches input
    if len(final_vol) != len(data):
        # Pad with first/last values if needed
        result = np.full(len(data), final_vol.iloc[-1])
        result[:len(final_vol)] = final_vol.values
    else:
        result = final_vol.values
    
    return result


class VolatilityMutingLayer:
    """
    Enhanced volatility muting layer with adaptive activation function
    Low volatility = higher muting factors (amplify reliable signals)
    High volatility = lower muting factors (dampen unreliable signals)
    """
    def __init__(self, input_size, volatility_sensitivity=1.0, base_muting_factor=1.0):
        # One-to-one mapping weights
        self.weights = np.eye(input_size) + np.random.randn(input_size, input_size) * 0.01
        self.bias = np.zeros(input_size)
        
        # Volatility-specific parameters
        self.volatility_sensitivity = volatility_sensitivity
        self.base_muting_factor = base_muting_factor
        
        # Track volatility statistics for normalization
        self.volatility_stats = {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0}
        
        # Adaptive volatility activation function
        self.volatility_activation = AdaptivePiecewiseActivation(
            num_pieces=7,  # Odd number for center symmetry
            polynomial_degree=2,
            init_range=(-3, 3),  # Normalized volatility range
            reg_lambda=0.001
        )
        
        # Initialize with symmetric volatility-appropriate behavior
        self._initialize_symmetric_volatility_activation()
        
    def _initialize_symmetric_volatility_activation(self):
        """
        Initialize the volatility activation with symmetric behavior:
        - Center piece (low volatility): High amplification
        - Outer pieces (high volatility): Low amplification/dampening
        """
        num_pieces = self.volatility_activation.num_pieces
        center_idx = num_pieces // 2  # Center piece index
        
        # Center piece: Low volatility regime (x ≈ 0) → High muting factors
        self.volatility_activation.coefficients[center_idx, 0] = 1.5  # High baseline amplification
        self.volatility_activation.coefficients[center_idx, 1] = 0.0  # No linear term (symmetric)
        self.volatility_activation.coefficients[center_idx, 2] = -0.15  # Negative quadratic (inverted parabola)
        
        # Symmetric initialization for outer pieces
        for offset in range(1, center_idx + 1):
            left_idx = center_idx - offset
            right_idx = center_idx + offset
            
            # Decreasing amplification as we move away from center
            base_amp = max(1.5 - 0.2 * offset, 0.3)  # Decreasing baseline, min 0.3
            linear_coef = 0.0  # Keep linear term zero for symmetry
            quad_coef = -0.05 * offset  # Negative quadratic, stronger for outer pieces
            
            # Left piece (negative volatility deviations)
            if left_idx >= 0:
                self.volatility_activation.coefficients[left_idx, 0] = base_amp
                self.volatility_activation.coefficients[left_idx, 1] = linear_coef
                self.volatility_activation.coefficients[left_idx, 2] = quad_coef
            
            # Right piece (positive volatility deviations) - mirror of left
            if right_idx < num_pieces:
                self.volatility_activation.coefficients[right_idx, 0] = base_amp
                self.volatility_activation.coefficients[right_idx, 1] = linear_coef
                self.volatility_activation.coefficients[right_idx, 2] = quad_coef
        
        # Ensure outermost pieces provide minimum dampening (don't go below 0.2)
        for i in [0, num_pieces - 1]:
            self.volatility_activation.coefficients[i, 0] = max(
                self.volatility_activation.coefficients[i, 0], 0.2
            )
    
    def update_volatility_stats(self, implied_vol_batch):
        """Update comprehensive volatility statistics with momentum"""
        if len(implied_vol_batch) == 0:
            return
            
        current_mean = np.mean(implied_vol_batch)
        current_std = np.std(implied_vol_batch)
        current_min = np.min(implied_vol_batch)
        current_max = np.max(implied_vol_batch)
        
        # Exponential moving average for stability
        alpha = 0.05  # Slower adaptation for stability
        self.volatility_stats['mean'] = (1 - alpha) * self.volatility_stats['mean'] + alpha * current_mean
        self.volatility_stats['std'] = (1 - alpha) * self.volatility_stats['std'] + alpha * current_std
        self.volatility_stats['min'] = (1 - alpha) * self.volatility_stats['min'] + alpha * current_min
        self.volatility_stats['max'] = (1 - alpha) * self.volatility_stats['max'] + alpha * current_max
        
        # Ensure std doesn't become too small
        self.volatility_stats['std'] = max(self.volatility_stats['std'], 0.01)
    
    def compute_muting_factors(self, implied_volatility):
        """
        Compute adaptive muting factors using the learnable activation:
        - Normalize volatility deviations from mean
        - Apply adaptive activation function
        - Low volatility → High muting factors (amplify)
        - High volatility → Low muting factors (dampen)
        """
        # Center volatility around its mean (symmetric about 0)
        centered_vol = implied_volatility - self.volatility_stats['mean']
        
        # Normalize by standard deviation
        normalized_vol = centered_vol / (self.volatility_stats['std'] + 1e-8)
        
        # Clip to reasonable range for stability
        normalized_vol = np.clip(normalized_vol, -3, 3)
        
        # Apply adaptive activation function
        # The function learns to map normalized volatility to muting factors
        muting_factors = self.volatility_activation.forward(normalized_vol)
        
        # Ensure muting factors are in reasonable bounds
        # High end: amplify low-vol signals up to 2.5x for strong signals
        # Low end: dampen high-vol signals down to 0.15x for very noisy periods
        muting_factors = np.clip(muting_factors, 0.15, 2.5)
        
        return muting_factors
    
    def forward(self, x, implied_volatility):
        """Forward pass with adaptive volatility-based muting"""
        batch_size = x.shape[0]
        
        # Update volatility statistics
        self.update_volatility_stats(implied_volatility)
        
        # Standard linear transformation
        linear_output = np.dot(x, self.weights) + self.bias
        
        # Compute adaptive muting factors
        muting_factors = self.compute_muting_factors(implied_volatility)
        
        # Apply muting (broadcasting across features)
        muted_output = linear_output * muting_factors.reshape(-1, 1)
        
        return muted_output
    
    def backward(self, x, output_grad, implied_volatility, learning_rate):
        """Backward pass with adaptive volatility activation learning"""
        batch_size = x.shape[0]
        
        # Compute normalized volatility for activation function
        centered_vol = implied_volatility - self.volatility_stats['mean']
        normalized_vol = centered_vol / (self.volatility_stats['std'] + 1e-8)
        normalized_vol = np.clip(normalized_vol, -3, 3)
        
        # Get current muting factors
        muting_factors = self.compute_muting_factors(implied_volatility)
        
        # Standard parameter gradients
        d_weights = np.dot(x.T, output_grad * muting_factors.reshape(-1, 1)) / batch_size
        d_bias = np.mean(output_grad * muting_factors.reshape(-1, 1), axis=0)
        
        # Update standard parameters
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        
        # Compute gradients for the adaptive activation function
        linear_output = np.dot(x, self.weights) + self.bias
        
        # The gradient of the loss w.r.t. muting factors
        # Sum across features since muting factor affects all features equally
        muting_grad = np.sum(output_grad * linear_output, axis=1)
        
        # Update the adaptive activation function
        # Use slower learning rate for activation to maintain stability
        activation_lr = learning_rate * 0.02
        self.volatility_activation.update_parameters(
            normalized_vol, 
            muting_grad, 
            activation_lr
        )
        
        # Optional: Maintain symmetry in activation function
        if hasattr(self, '_maintain_symmetry') and self._maintain_symmetry:
            self._enforce_activation_symmetry()
        
        # Propagate gradients backward
        input_grad = np.dot(output_grad * muting_factors.reshape(-1, 1), self.weights.T)
        
        return input_grad
    
    def _enforce_activation_symmetry(self):
        """
        Optionally enforce symmetry in the activation function coefficients
        This ensures the function remains symmetric about x=0
        """
        num_pieces = self.volatility_activation.num_pieces
        center_idx = num_pieces // 2
        
        # Force center piece to have zero linear term (symmetry requirement)
        self.volatility_activation.coefficients[center_idx, 1] = 0.0
        
        # Mirror coefficients between left and right pieces
        for offset in range(1, center_idx + 1):
            left_idx = center_idx - offset
            right_idx = center_idx + offset
            
            if left_idx >= 0 and right_idx < num_pieces:
                # Average the coefficients to maintain symmetry
                avg_const = (self.volatility_activation.coefficients[left_idx, 0] + 
                           self.volatility_activation.coefficients[right_idx, 0]) / 2
                avg_quad = (self.volatility_activation.coefficients[left_idx, 2] + 
                          self.volatility_activation.coefficients[right_idx, 2]) / 2
                
                # Apply averaged values (maintaining symmetry)
                self.volatility_activation.coefficients[left_idx, 0] = avg_const
                self.volatility_activation.coefficients[right_idx, 0] = avg_const
                self.volatility_activation.coefficients[left_idx, 1] = 0.0  # Force zero linear
                self.volatility_activation.coefficients[right_idx, 1] = 0.0  # Force zero linear  
                self.volatility_activation.coefficients[left_idx, 2] = avg_quad
                self.volatility_activation.coefficients[right_idx, 2] = avg_quad
    
    def get_activation_function_info(self):
        """
        Get information about the learned activation function
        Useful for debugging and understanding what the network has learned
        """
        return {
            'coefficients': self.volatility_activation.coefficients.copy(),
            'breakpoints': self.volatility_activation.breakpoints.copy(),
            'volatility_stats': self.volatility_stats.copy(),
            'num_pieces': self.volatility_activation.num_pieces
        }
    
    def set_symmetry_enforcement(self, enforce=True):
        """Enable or disable symmetry enforcement in the activation function"""
        self._maintain_symmetry = enforce

class AdaptivePiecewiseActivation:
    """
    Enhanced piecewise activation function with 5 pieces and proper regularization.
    Each piece can be a polynomial function with learnable coefficients.
    """
    def __init__(self, input_dim=1, num_pieces=5, polynomial_degree=2, init_range=(-3, 3),
                 reg_lambda=0.001):  # MUCH smaller regularization weight
        # Initialize boundaries (sorted points that separate the pieces)
        self.num_pieces = num_pieces
        self.num_boundaries = num_pieces - 1
        
        # Evenly distribute initial boundaries across the specified range
        if self.num_boundaries > 0:
            self.boundaries = np.linspace(init_range[0], init_range[1], self.num_boundaries + 2)[1:-1]
        else:
            self.boundaries = np.array([])
        
        # Initialize polynomial coefficients for each piece
        self.polynomial_degree = polynomial_degree
        self.coefficients = np.zeros((num_pieces, polynomial_degree + 1))
        
        # Initialize with more diverse segments for better expressiveness
        for i in range(num_pieces):
            if i == num_pieces // 2:  # Middle piece as identity-like
                self.coefficients[i, 1] = 1.0  # x^1 coefficient
                self.coefficients[i, 0] = 0.0  # constant term
            elif i < num_pieces // 2:  # Left pieces - handle negative values
                self.coefficients[i, 1] = 0.6 + 0.2 * i  # Increasing slope
                self.coefficients[i, 0] = -0.1 * (num_pieces // 2 - i)  # Negative offset
            else:  # Right pieces - handle positive values  
                self.coefficients[i, 1] = 0.8 + 0.2 * (i - num_pieces // 2)  # Increasing slope
                self.coefficients[i, 0] = 0.1 * (i - num_pieces // 2)  # Positive offset
        
        # Learning rate scaling factors
        self.boundary_lr_scale = 0.1
        self.coefficient_lr_scale = 0.2
        
        # Smoothing for piece transitions
        self.smoothing_factor = 4.0
        
        # History tracking
        self.boundary_history = [self.boundaries.copy()]
        self.coefficient_history = [self.coefficients.copy()]
        
        # For adaptive learning rates
        self.piece_usage_count = np.zeros(num_pieces)
        self.boundary_stress = np.zeros(self.num_boundaries)
        
        self.reg_lambda = reg_lambda  # Much smaller regularization weight
        
        # Gradient clipping and momentum for stability
        self.coefficient_momentum = np.zeros((num_pieces, polynomial_degree + 1))
        self.boundary_momentum = np.zeros(self.num_boundaries) if self.num_boundaries > 0 else np.array([])
        self.momentum_decay = 0.9
        
    def _get_piece_weights(self, x):
        """Calculate smooth weights for each piece at input x"""
        x = np.clip(x, -15, 15)
        x = x.reshape(-1, 1)
        batch_size = x.shape[0]
        weights = np.ones((batch_size, self.num_pieces))
        
        for i, boundary in enumerate(self.boundaries):
            transition = sigmoid(self.smoothing_factor * (x - boundary)).flatten()
            
            for p in range(i + 1):
                weights[:, p] *= (1 - transition)
            for p in range(i + 1, self.num_pieces):
                weights[:, p] *= transition
        
        epsilon = 1e-12
        return weights / (weights.sum(axis=1, keepdims=True) + epsilon)

    def _evaluate_polynomial(self, x, coeffs):
        """Evaluate polynomial with given coefficients at x"""
        x_clipped = np.clip(x, -8, 8)
        result = np.zeros_like(x_clipped)
        
        for i, coef in enumerate(coeffs):
            if i == 0:
                term = coef
            elif i == 1:
                term = coef * x_clipped
            else:
                term = coef * np.power(x_clipped, min(i, 3))
                
            result += term
            
        return np.clip(result.ravel(), -50, 50)

    def forward(self, x):
        """Forward pass with proper dimension handling"""
        x = np.asarray(x).reshape(-1, 1)
        batch_size = x.shape[0]
        piece_weights = self._get_piece_weights(x)
        
        piece_outputs = np.zeros((batch_size, self.num_pieces))
        for p in range(self.num_pieces):
            piece_output = self._evaluate_polynomial(x, self.coefficients[p])
            piece_outputs[:, p] = piece_output
            
        output = np.sum(piece_weights * piece_outputs, axis=1)
        return np.clip(output, -50, 50)
    
    def get_derivative(self, x):
        """Calculate derivative of the activation function at x"""
        x = np.clip(x, -15, 15)
        # Flatten x to 1D array
        x_flat = x.flatten()
        batch_size = x_flat.shape[0]
        piece_weights = self._get_piece_weights(x)  # This expects 2D input but we'll handle differently

        piece_derivatives = np.zeros((batch_size, self.num_pieces))
        
        for p in range(self.num_pieces):
            for i in range(1, self.polynomial_degree + 1):
                safe_power = min(i-1, 3)
                # Use flattened x for calculations
                term = i * self.coefficients[p, i] * (np.clip(x_flat, -8, 8) ** safe_power)
                piece_derivatives[:, p] += term
        
        # Rest of the function remains the same but uses x_flat
        weight_derivatives = np.zeros((batch_size, self.num_pieces))
        
        for i, boundary in enumerate(self.boundaries):
            sig = sigmoid(self.smoothing_factor * (x_flat - boundary))
            sig_derivative = self.smoothing_factor * sig * (1 - sig)
            
            for p in range(self.num_pieces):
                if p <= i:
                    weight_derivatives[:, p] -= sig_derivative
                else:
                    weight_derivatives[:, p] += sig_derivative
        
        piece_output_values = np.zeros((batch_size, self.num_pieces))
        for p in range(self.num_pieces):
            piece_output_values[:, p] = self._evaluate_polynomial(x_flat, self.coefficients[p])
        
        derivative_from_weights = np.sum(weight_derivatives * piece_output_values, axis=1)
        derivative_from_pieces = np.sum(piece_weights * piece_derivatives, axis=1)
        
        total_derivative = derivative_from_weights + derivative_from_pieces
        return np.clip(total_derivative, -20, 20)
    
    def compute_coefficient_gradients(self, x, output_grad):
        """Compute gradients for polynomial coefficients"""
        x = np.clip(x, -15, 15)
        output_grad = np.clip(output_grad, -20, 20)
        
        batch_size = x.shape[0]
        piece_weights = self._get_piece_weights(x)
        
        coefficient_grads = np.zeros_like(self.coefficients)
        
        for p in range(self.num_pieces):
            for power in range(self.polynomial_degree + 1):
                if power == 0:
                    x_power = np.ones_like(x)
                elif power == 1:
                    x_power = x
                else:
                    max_safe_power = min(power, 3)
                    x_power = np.power(np.clip(x, -5, 5), max_safe_power)
                
                weighted_grad = output_grad * piece_weights[:, p] * x_power.flatten()
                weighted_grad = np.clip(weighted_grad, -200, 200)
                coefficient_grads[p, power] = np.sum(weighted_grad)
                
                # Add L2 regularization gradient (properly weighted)
                coefficient_grads[p, power] += self.reg_lambda * self.coefficients[p, power]
        
        coefficient_grads = np.clip(coefficient_grads, -2.0, 2.0)
        return coefficient_grads
    
    def compute_boundary_gradients(self, x, output_grad):
        """Compute gradients for boundaries"""
        if self.num_boundaries == 0:
            return np.array([])
        
        x = np.clip(x, -15, 15)
        output_grad = np.clip(output_grad, -20, 20)
            
        batch_size = x.shape[0]
        boundary_grads = np.zeros(self.num_boundaries)
        
        piece_outputs = np.zeros((batch_size, self.num_pieces))
        for p in range(self.num_pieces):
            piece_outputs[:, p] = self._evaluate_polynomial(x, self.coefficients[p])
        
        for b_idx, boundary in enumerate(self.boundaries):
            sig = sigmoid(self.smoothing_factor * (x - boundary))
            sig_derivative = self.smoothing_factor * sig * (1 - sig)
            
            boundary_effect = np.zeros(batch_size)
            
            for i in range(batch_size):
                left_effect = np.sum(piece_outputs[i, :b_idx + 1])
                right_effect = np.sum(piece_outputs[i, b_idx + 1:])
                boundary_effect[i] = (left_effect - right_effect) * sig_derivative[i]
            
            boundary_effect = np.clip(boundary_effect, -20, 20)
            boundary_grads[b_idx] = np.sum(output_grad * boundary_effect)
            
            # Add regularization for boundaries too
            boundary_grads[b_idx] += self.reg_lambda * boundary
            
            self.boundary_stress[b_idx] = np.abs(boundary_grads[b_idx])
        
        boundary_grads = np.clip(boundary_grads, -2.0, 2.0)
        return boundary_grads
    
    def update_parameters(self, x, output_grad, global_learning_rate):
        """Parameter updates with proper regularization"""
        output_grad = np.clip(output_grad, -5, 5)
        
        coefficient_grads = self.compute_coefficient_gradients(x, output_grad)
        boundary_grads = self.compute_boundary_gradients(x, output_grad)
        
        # Apply momentum
        self.coefficient_momentum = (self.momentum_decay * self.coefficient_momentum + 
                                   (1 - self.momentum_decay) * coefficient_grads)
        
        # Update coefficients
        for p in range(self.num_pieces):
            effective_lr = global_learning_rate * self.coefficient_lr_scale
            self.coefficients[p] -= effective_lr * self.coefficient_momentum[p]
            self.coefficients[p] = np.clip(self.coefficients[p], -3.0, 3.0)
        
        # Update boundaries
        if self.num_boundaries > 0:
            self.boundary_momentum = (self.momentum_decay * self.boundary_momentum + 
                                    (1 - self.momentum_decay) * boundary_grads)
            
            for b_idx in range(self.num_boundaries):
                effective_lr = global_learning_rate * self.boundary_lr_scale
                self.boundaries[b_idx] -= effective_lr * self.boundary_momentum[b_idx]
            
            self.boundaries = np.sort(self.boundaries)
            # Prevent boundaries from collapsing
            for i in range(len(self.boundaries) - 1):
                if self.boundaries[i+1] - self.boundaries[i] < 0.1:
                    self.boundaries[i+1] = self.boundaries[i] + 0.1
        
        # FIXED: Calculate total regularization loss properly
        coeff_reg = np.sum(self.coefficients**2)
        boundary_reg = np.sum(self.boundaries**2) if len(self.boundaries) > 0 else 0
        reg_loss = 0.5 * self.reg_lambda * (coeff_reg + boundary_reg)
        
        return {
            'coefficient_grads': coefficient_grads,
            'boundary_grads': boundary_grads,
            'reg_loss': reg_loss
        }
        
    def get_regularization_loss(self):
        """Calculate the L2 regularization loss"""
        coeff_reg = np.sum(self.coefficients**2)
        boundary_reg = np.sum(self.boundaries**2) if len(self.boundaries) > 0 else 0
        return 0.5 * self.reg_lambda * (coeff_reg + boundary_reg)
    def get_state(self):
        return {
            'boundaries': self.boundaries.copy(),
            'coefficients': self.coefficients.copy(),
            'boundary_history': [bh.copy() for bh in self.boundary_history],
            'coefficient_history': [ch.copy() for ch in self.coefficient_history],
            'piece_usage_count': self.piece_usage_count.copy(),
            'boundary_stress': self.boundary_stress.copy(),
            'coefficient_momentum': self.coefficient_momentum.copy(),
            'boundary_momentum': self.boundary_momentum.copy(),
            'reg_lambda': self.reg_lambda
        }
    
    def set_state(self, state):
        self.boundaries = state['boundaries']
        self.coefficients = state['coefficients']
        self.boundary_history = state['boundary_history']
        self.coefficient_history = state['coefficient_history']
        self.piece_usage_count = state['piece_usage_count']
        self.boundary_stress = state['boundary_stress']
        self.coefficient_momentum = state['coefficient_momentum']
        self.boundary_momentum = state['boundary_momentum']
        self.reg_lambda = state['reg_lambda']


class TimeWeightedNeuralNetwork:
    """Neural network with time-weighted learning and adaptive piecewise activations"""
    
    def __init__(self, input_size, hidden_sizes, output_size, time_decay_rate=0.95, use_volatility_muting=True):
        self.layers = []
        self.activations = []
        self.time_weights = []  # Time-based importance weights for each neuron
        self.time_decay_rate = time_decay_rate
        
        # Build network architecture
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            # Weight matrix for this layer
            weights = np.random.randn(prev_size, hidden_size) * np.sqrt(2.0 / prev_size)
            bias = np.zeros(hidden_size)
            self.layers.append({'weights': weights, 'bias': bias})
            
            # Adaptive piecewise activation for each neuron in this layer
            layer_activations = []
            for j in range(hidden_size):
                activation = AdaptivePiecewiseActivation(
                    num_pieces=5, 
                    polynomial_degree=2, 
                    reg_lambda=0.0001
                )
                layer_activations.append(activation)
            self.activations.append(layer_activations)
            
            # Time weights for each neuron (initialized to 1.0)
            self.time_weights.append(np.ones(hidden_size))
            
            prev_size = hidden_size
        
        # Output layer
        output_weights = np.random.randn(prev_size, output_size) * np.sqrt(2.0 / prev_size)
        output_bias = np.zeros(output_size)
        self.layers.append({'weights': output_weights, 'bias': output_bias})
        
        # Track layer outputs for backprop
        self.layer_outputs = []
        self.pre_activation_values = []

        # Add volatility muting layer after first hidden layer if enabled
        self.use_volatility_muting = use_volatility_muting
        if use_volatility_muting and len(hidden_sizes) > 0:
            self.volatility_muting_layer = VolatilityMutingLayer(
                hidden_sizes[0],  # Size of first hidden layer
                volatility_sensitivity=1.5,
                base_muting_factor=0.8
            )
        else:
            self.volatility_muting_layer = None
        
    def compute_time_weights(self, time_indices, max_time_index):
        """Compute time-based weights where recent data gets higher weight"""
        # Normalize time indices to [0, 1] range
        normalized_time = time_indices / max_time_index
        
        # Exponential weighting favoring recent data
        # Use a gentler weighting to avoid gradient vanishing
        time_weights = np.power(self.time_decay_rate, (1 - normalized_time))
        
        # Apply minimum weight threshold to prevent complete gradient vanishing
        min_weight = 0.1
        time_weights = np.maximum(time_weights, min_weight)
        
        # Apply gradient scaling to compensate for reduced weights on older data
        # This helps address your concern about small gradients
        gradient_scale = 1.0 / np.sqrt(time_weights + 1e-8)
        
        return time_weights, gradient_scale
    
    def forward(self, X, time_indices=None, max_time_index=None, implied_volatility=None):
        """Forward pass with volatility muting"""
        self.layer_outputs = [X]
        self.pre_activation_values = []
        
        current_input = X
        
        for layer_idx, (layer, layer_activations) in enumerate(zip(self.layers[:-1], self.activations)):
            # Linear transformation
            z = np.dot(current_input, layer['weights']) + layer['bias']
            self.pre_activation_values.append(z)
            
            # Apply adaptive piecewise activation
            activated_output = np.zeros_like(z)
            for neuron_idx, activation_func in enumerate(layer_activations):
                neuron_input = z[:, neuron_idx]
                activated_output[:, neuron_idx] = activation_func.forward(neuron_input)
            
            # Apply volatility muting after first hidden layer
            if (layer_idx == 0 and self.volatility_muting_layer is not None and 
                implied_volatility is not None):
                activated_output = self.volatility_muting_layer.forward(activated_output, implied_volatility)
            
            # Apply time weighting
            if time_indices is not None and max_time_index is not None:
                time_weights, _ = self.compute_time_weights(time_indices, max_time_index)
                for sample_idx in range(activated_output.shape[0]):
                    activated_output[sample_idx] *= time_weights[sample_idx]
            
            self.layer_outputs.append(activated_output)
            current_input = activated_output
        
        # Output layer (linear)
        output = np.dot(current_input, self.layers[-1]['weights']) + self.layers[-1]['bias']
        self.layer_outputs.append(output)
        
        return output
        
    # Modified backward method - add this section after the existing backward logic
    def backward(self, X, y, time_indices=None, max_time_index=None, learning_rate=0.001, implied_volatility=None):
        """Backward pass with volatility muting gradients"""
        batch_size = X.shape[0]
        
        # Compute time weights and gradient scaling
        if time_indices is not None and max_time_index is not None:
            time_weights, gradient_scale = self.compute_time_weights(time_indices, max_time_index)
        else:
            time_weights = np.ones(batch_size)
            gradient_scale = np.ones(batch_size)
        
        # Output layer gradient
        output_error = self.layer_outputs[-1] - y
        
        # Apply time weighting to error
        for sample_idx in range(batch_size):
            output_error[sample_idx] *= time_weights[sample_idx]
            output_error[sample_idx] *= gradient_scale[sample_idx]
        
        # Update output layer
        d_output_weights = np.dot(self.layer_outputs[-2].T, output_error) / batch_size
        d_output_bias = np.mean(output_error, axis=0)
        
        self.layers[-1]['weights'] -= learning_rate * d_output_weights
        self.layers[-1]['bias'] -= learning_rate * d_output_bias
        
        # Backpropagate through hidden layers
        current_error = np.dot(output_error, self.layers[-1]['weights'].T)
        
        for layer_idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[layer_idx]
            layer_activations = self.activations[layer_idx]
            pre_activation = self.pre_activation_values[layer_idx]
            
            # Handle volatility muting layer gradients (after first hidden layer)
            if (layer_idx == 0 and self.volatility_muting_layer is not None and 
                implied_volatility is not None):
                # Get the input to volatility muting layer (activated output before muting)
                pre_muting_output = np.zeros_like(pre_activation)
                for neuron_idx, activation_func in enumerate(layer_activations):
                    neuron_input = pre_activation[:, neuron_idx]
                    pre_muting_output[:, neuron_idx] = activation_func.forward(neuron_input)
                
                # Backpropagate through volatility muting layer
                current_error = self.volatility_muting_layer.backward(
                    pre_muting_output, current_error, implied_volatility, learning_rate
                )
            
            # Compute activation derivatives and update activation parameters
            activation_derivatives = np.zeros_like(current_error)
            
            for neuron_idx, activation_func in enumerate(layer_activations):
                neuron_input = pre_activation[:, neuron_idx]
                neuron_error = current_error[:, neuron_idx]
                
                # Apply time weighting and gradient scaling to neuron error
                weighted_error = neuron_error * time_weights * gradient_scale
                
                # Update activation function parameters
                activation_func.update_parameters(
                    neuron_input, 
                    weighted_error, 
                    learning_rate * 0.1
                )
                
                # Get derivative for backprop
                activation_derivatives[:, neuron_idx] = activation_func.get_derivative(neuron_input)
            
            # Apply activation derivatives
            current_error *= activation_derivatives
            
            # Update layer weights and biases
            if layer_idx > 0:
                d_weights = np.dot(self.layer_outputs[layer_idx].T, current_error) / batch_size
                d_bias = np.mean(current_error, axis=0)
                
                layer['weights'] -= learning_rate * d_weights
                layer['bias'] -= learning_rate * d_bias
                
                current_error = np.dot(current_error, layer['weights'].T)
        
    def train(self, X, y, time_indices, epochs=100, learning_rate=0.001, batch_size=32, implied_volatility=None):
        """Train with implied volatility support"""
        max_time_index = np.max(time_indices)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                batch_X = X[i:end_idx]
                batch_y = y[i:end_idx]
                batch_time_indices = time_indices[i:end_idx]
                batch_implied_vol = implied_volatility[i:end_idx] if implied_volatility is not None else None
                
                # Forward pass
                predictions = self.forward(batch_X, batch_time_indices, max_time_index, batch_implied_vol)
                
                # Compute loss
                time_weights, _ = self.compute_time_weights(batch_time_indices, max_time_index)
                batch_loss = np.mean(time_weights.reshape(-1, 1) * (predictions - batch_y) ** 2)
                epoch_loss += batch_loss
                num_batches += 1
                
                # Backward pass
                self.backward(batch_X, batch_y, batch_time_indices, max_time_index, learning_rate, batch_implied_vol)
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return losses
    def predict(self, X):
        """Make predictions without time weighting"""
        return self.forward(X)

    def get_state(self):
        layers_state = []
        for layer in self.layers:
            layers_state.append({
                'weights': layer['weights'].copy(),
                'bias': layer['bias'].copy()
            })
        
        activations_state = []
        for layer_activations in self.activations:
            layer_act_state = []
            for act in layer_activations:
                layer_act_state.append(act.get_state())
            activations_state.append(layer_act_state)
        
        time_weights_state = [tw.copy() for tw in self.time_weights]
        
        return {
            'layers': layers_state,
            'activations': activations_state,
            'time_weights': time_weights_state
        }
    
    def set_state(self, state):
        self.layers = []
        for layer_state in state['layers']:
            self.layers.append({
                'weights': layer_state['weights'],
                'bias': layer_state['bias']
            })
        
        self.activations = []
        for layer_act_state in state['activations']:
            layer_activations = []
            for act_state in layer_act_state:
                act = AdaptivePiecewiseActivation()
                act.set_state(act_state)
                layer_activations.append(act)
            self.activations.append(layer_activations)
        
        self.time_weights = state['time_weights']

class VolatilityForecaster:
    def __init__(self, data, model_path='vol_forecaster_model.pkl'):
        self.data = data
        self.model = None
        self.model_path = model_path
        self.best_decay_rate = None
        self.training_time = 0
        self.energy_consumed = 0
        self.last_train_date = None
        self.feature_columns = ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'implied_vol']
    
    def prepare_features(self, last_n_years=5):
        """Create time series features from OHLC data"""
        df = self.data.copy()
        
        # Calculate returns
        df['ret_1'] = df['Close'].pct_change()
        df['ret_5'] = df['Close'].pct_change(5)
        df['ret_10'] = df['Close'].pct_change(10)
        df['ret_20'] = df['Close'].pct_change(20)
        
        # Calculate implied volatility
        df['implied_vol'] = calculate_implied_volatility(df)
        
        # Target: next day return
        df['target'] = df['ret_1'].shift(-1)
        
        # Filter last N years
        end_date = df.index.max()
        start_date = end_date - pd.DateOffset(years=last_n_years)
        df = df.loc[start_date:end_date].dropna()
        
        # Create time indices (0 = oldest, T = newest)
        time_indices = np.arange(len(df))
        
        return df[self.feature_columns], df['target'], time_indices
    
    def train_model(self, decay_rates=[0.85, 0.90, 0.95, 0.97, 0.99], 
                   epochs=50, learning_rate=0.001, batch_size=64):
        start_time = time.time()
        
        # Prepare data
        X, y, time_indices = self.prepare_features(last_n_years=5)
        max_time = len(X) - 1
        
        # Time-based split (80-10-10)
        train_size = int(0.8 * len(X))
        cv_size = int(0.1 * len(X))
        
        X_train, y_train, ti_train = X[:train_size], y[:train_size], time_indices[:train_size]
        X_cv, y_cv, ti_cv = X[train_size:train_size+cv_size], y[train_size:train_size+cv_size], time_indices[train_size:train_size+cv_size]
        X_test, y_test, ti_test = X[train_size+cv_size:], y[train_size+cv_size:], time_indices[train_size+cv_size:]
        
        # Cross-validation for decay rate
        best_loss = float('inf')
        best_decay = decay_rates[0]
        
        for decay_rate in decay_rates:
            model = TimeWeightedNeuralNetwork(
                input_size=5,
                hidden_sizes=[20, 10],
                output_size=1,
                time_decay_rate=decay_rate
            )
            
            # Train on training set
            model.train(
                X_train.values, y_train.values.reshape(-1, 1),
                ti_train, epochs=epochs, 
                learning_rate=learning_rate, batch_size=batch_size
            )
            
            # Evaluate on CV set
            predictions = model.forward(X_cv.values)
            loss = np.mean(predictions - y_cv.values.reshape(-1, 1))**2
            
            if loss < best_loss:
                best_loss = loss
                best_decay = decay_rate
                print(f"New best decay: {best_decay:.4f} (Loss: {loss:.6f})")
        
        self.best_decay_rate = best_decay
        print(f"Selected decay rate: {best_decay:.4f}")
        
        # Retrain on combined train + CV with best decay
        combined_X = np.vstack((X_train.values, X_cv.values))
        combined_y = np.vstack((y_train.values.reshape(-1, 1), y_cv.values.reshape(-1, 1)))
        combined_ti = np.concatenate((ti_train, ti_cv))
        
        final_model = TimeWeightedNeuralNetwork(
            input_size=5,
            hidden_sizes=[20, 10],
            output_size=1,
            time_decay_rate=best_decay
        )
        
        final_model.train(
            combined_X, combined_y,
            combined_ti, epochs=epochs,
            learning_rate=learning_rate, batch_size=batch_size
        )
        
        # Test set evaluation
        test_preds = final_model.forward(X_test.values)
        test_loss = np.mean(test_preds - y_test.values.reshape(-1, 1))**2
        print(f"Test Loss: {test_loss:.6f}")
        
        self.model = final_model
        self.last_train_date = datetime.now()
        
        # Calculate training metrics
        self.training_time = time.time() - start_time
        self.energy_consumed = (self.training_time / 3600) * POWER_CONSUMPTION_KW  # kWh
        
        # Save model
        self.save_model()
    
    def save_model(self):
        """Save model parameters and metadata"""
        model_data = {
            'model_state': self.model.get_state(),
            'decay_rate': self.best_decay_rate,
            'last_train_date': self.last_train_date,
            'training_time': self.training_time,
            'energy_consumed': self.energy_consumed,
            'feature_columns': self.feature_columns
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Load model parameters from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Reconstruct model
            self.model = TimeWeightedNeuralNetwork(5, [20, 10], 1)
            self.model.set_state(model_data['model_state'])
            self.best_decay_rate = model_data['decay_rate']
            self.last_train_date = model_data['last_train_date']
            self.training_time = model_data['training_time']
            self.energy_consumed = model_data['energy_consumed']
            self.feature_columns = model_data['feature_columns']
            return True
        except:
            return False
    
    def forecast_next_month(self, days=20):
        """Project returns for the next trading month"""
        if not self.model:
            raise ValueError("Model not initialized. Train or load model first")
        
        # Prepare most recent window
        _, _, time_indices = self.prepare_features()
        last_window = self.data.tail(20).copy()
        projections = []
        
        for _ in range(days):
            # Calculate features for current window
            current_close = last_window['Close'].iloc[-1]
            features = pd.DataFrame({
                'ret_1': [last_window['Close'].iloc[-1] / last_window['Close'].iloc[-2] - 1],
                'ret_5': [last_window['Close'].iloc[-1] / last_window['Close'].iloc[-6] - 1],
                'ret_10': [last_window['Close'].iloc[-1] / last_window['Close'].iloc[-11] - 1],
                'ret_20': [last_window['Close'].iloc[-1] / last_window['Close'].iloc[-21] - 1],
                'implied_vol': [calculate_implied_volatility(last_window)[-1]]
            })
            
            # Predict next return
            pred_return = self.model.forward(features.values)[0][0]
            next_close = current_close * (1 + pred_return)
            
            # Create new date (next business day)
            new_date = last_window.index[-1] + timedelta(days=1)
            while new_date.weekday() >= 5:  # Skip weekends
                new_date += timedelta(days=1)
            
            # Update window
            new_row = pd.DataFrame({'Close': next_close}, index=[new_date])
            last_window = pd.concat([last_window.iloc[1:], new_row])
            
            projections.append({
                'date': new_date,
                'predicted_close': next_close,
                'predicted_return': pred_return
            })
        
        return pd.DataFrame(projections)
    
    def run(self):
        """Main execution flow with user prompt"""
        if os.path.exists(self.model_path):
            response = input("Model found. Retrain? (y/n): ").strip().lower()
            if response == 'y':
                print("Retraining model...")
                self.train_model()
            else:
                print("Loading saved model...")
                if not self.load_model():
                    print("Error loading model. Retraining...")
                    self.train_model()
        else:
            print("No saved model found. Training new model...")
            self.train_model()
        
        # Show training metrics
        print(f"\nTraining Metrics:")
        print(f"- Time: {self.training_time:.2f} seconds")
        print(f"- Energy: {self.energy_consumed:.6f} kWh")
        print(f"- Last Train Date: {self.last_train_date}")
        
        # Generate projections
        projections = self.forecast_next_month()
        print("\nNext Month Projections:")
        print(projections[['date', 'predicted_close', 'predicted_return']])
        
        return projections
