#!/usr/bin/env python3
"""
Model Calibration Script
功能：
1. 加载已训练好的模型
2. 使用模型预测得到logits
3. 应用sigmoid得到预测概率p
4. 计算对数几率s = log(p/(1-p))
5. 使用区域化的可学习参数a和b进行校准：P = sigmoid(a*s + b)
6. 最小化负对数似然(NLL)损失训练a和b参数
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import json
from tqdm import tqdm
import logging
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegionCalibrationDataset(Dataset):
    """Dataset for region-based calibration"""
    
    def __init__(self, h5_file_path, lookup_table_path, lookback_length=30, 
                 forecast_horizon=7, max_samples=None):
        """
        Args:
            h5_file_path: Path to H5 file containing pixel data
            lookup_table_path: Path to lookup table file (row_col region_id format)
            lookback_length: Number of past days to use
            forecast_horizon: Number of future days to predict
            max_samples: Maximum number of samples to use (None for all)
        """
        self.h5_file_path = h5_file_path
        self.lookback_length = lookback_length
        self.forecast_horizon = forecast_horizon
        
        # Load lookup table
        self.pixel_to_region = self._load_lookup_table(lookup_table_path)
        self.regions = sorted(list(set(self.pixel_to_region.values())))
        self.num_regions = len(self.regions)
        
        logger.info(f"Found {self.num_regions} regions: {self.regions}")
        
        # Load H5 data and create samples
        self.samples = self._create_samples(max_samples)
        
        logger.info(f"Created {len(self.samples)} calibration samples")
    
    def _load_lookup_table(self, lookup_table_path):
        """Load lookup table mapping pixels to regions"""
        pixel_to_region = {}
        
        with open(lookup_table_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        pixel_coord = parts[0]  # row_col format
                        region_id = int(parts[1])
                        pixel_to_region[pixel_coord] = region_id
        
        logger.info(f"Loaded {len(pixel_to_region)} pixel-region mappings")
        return pixel_to_region
    
    def _create_samples(self, max_samples):
        """Create calibration samples from H5 data"""
        samples = []
        
        with h5py.File(self.h5_file_path, 'r') as f:
            pixel_datasets = list(f.keys())
            
            for pixel_name in tqdm(pixel_datasets, desc="Creating samples"):
                # Parse pixel coordinates
                if '_' in pixel_name:
                    parts = pixel_name.split('_')
                    if len(parts) >= 2:
                        row, col = int(parts[0]), int(parts[1])
                        pixel_coord = f"{row}_{col}"
                        
                        # Check if pixel has region mapping
                        if pixel_coord in self.pixel_to_region:
                            region_id = self.pixel_to_region[pixel_coord]
                            
                            # Load pixel data
                            pixel_data = f[pixel_name][:]  # shape: (channels, time_steps)
                            
                            # Create time windows
                            total_time = pixel_data.shape[1]
                            for start_idx in range(self.lookback_length, 
                                                 total_time - self.forecast_horizon + 1):
                                
                                # Past data (input)
                                past_data = pixel_data[:, start_idx-self.lookback_length:start_idx]
                                
                                # Future data (target)
                                future_data = pixel_data[:, start_idx:start_idx+self.forecast_horizon]
                                
                                # Use FIRMS channel (channel 0) as target
                                target = future_data[0, :]  # shape: (forecast_horizon,)
                                
                                # Check if target has valid data
                                if np.any(target > 0):  # At least one day has fire
                                    samples.append({
                                        'past_data': past_data,
                                        'target': target,
                                        'region_id': region_id,
                                        'pixel_coord': pixel_coord
                                    })
                                    
                                    if max_samples and len(samples) >= max_samples:
                                        break
                
                if max_samples and len(samples) >= max_samples:
                    break
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        past_data = torch.FloatTensor(sample['past_data'])
        target = torch.FloatTensor(sample['target'])
        region_id = torch.LongTensor([sample['region_id']])
        
        return {
            'past_data': past_data,
            'target': target,
            'region_id': region_id,
            'pixel_coord': sample['pixel_coord']
        }


class RegionCalibrationModel(nn.Module):
    """Region-based calibration model"""
    
    def __init__(self, num_regions, base_model, device='cuda'):
        """
        Args:
            num_regions: Number of regions
            base_model: Pre-trained base model
            device: Device to use
        """
        super().__init__()
        self.num_regions = num_regions
        self.base_model = base_model
        self.device = device
        
        # Region-specific calibration parameters
        self.region_a = nn.Parameter(torch.ones(num_regions))  # slope parameter
        self.region_b = nn.Parameter(torch.zeros(num_regions))  # intercept parameter
        
        # Move to device
        self.to(device)
        self.base_model.to(device)
        
        logger.info(f"Initialized calibration model for {num_regions} regions")
        logger.info(f"Initial a parameters: {self.region_a.data}")
        logger.info(f"Initial b parameters: {self.region_b.data}")
    
    def forward(self, past_data, region_ids):
        """
        Forward pass through calibration model
        Args:
            past_data: Input data [batch_size, channels, time_steps]
            region_ids: Region IDs [batch_size, 1]
        Returns:
            calibrated_probabilities: [batch_size, forecast_horizon]
        """
        # Get base model predictions (logits)
        with torch.no_grad():
            base_logits = self.base_model(past_data)  # [batch_size, forecast_horizon]
        
        # Convert logits to probabilities
        base_probs = torch.sigmoid(base_logits)
        
        # Calculate log-odds: s = log(p / (1-p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        base_probs_clipped = torch.clamp(base_probs, epsilon, 1 - epsilon)
        log_odds = torch.log(base_probs_clipped / (1 - base_probs_clipped))
        
        # Get region-specific parameters
        region_a = self.region_a[region_ids.squeeze()]  # [batch_size]
        region_b = self.region_b[region_ids.squeeze()]  # [batch_size]
        
        # Apply calibration: P = sigmoid(a * s + b)
        calibrated_logits = region_a.unsqueeze(1) * log_odds + region_b.unsqueeze(1)
        calibrated_probs = torch.sigmoid(calibrated_logits)
        
        return calibrated_probs, base_probs


def load_base_model(model_path, device='cuda'):
    """Load pre-trained base model"""
    logger.info(f"Loading base model from {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model instance (you may need to adjust this based on your model architecture)
    # This is a placeholder - replace with your actual model class
    from model_adapter_unified import UnifiedModelAdapter
    model = UnifiedModelAdapter(model_path, device=device)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info("Base model loaded successfully")
    return model


def train_calibration_model(calibration_model, dataloader, num_epochs=100, lr=0.01):
    """Train calibration parameters"""
    logger.info("Starting calibration training")
    
    optimizer = optim.Adam(calibration_model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binary Cross Entropy for NLL
    
    calibration_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            past_data = batch['past_data'].to(calibration_model.device)
            target = batch['target'].to(calibration_model.device)
            region_ids = batch['region_id'].to(calibration_model.device)
            
            # Forward pass
            calibrated_probs, base_probs = calibration_model(past_data, region_ids)
            
            # Calculate loss (NLL)
            loss = criterion(calibrated_probs, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Print calibration parameters
        if (epoch + 1) % 10 == 0:
            logger.info(f"Calibration parameters after epoch {epoch+1}:")
            for i, region_id in enumerate(calibration_model.regions):
                logger.info(f"  Region {region_id}: a={calibration_model.region_a[i].item():.4f}, "
                          f"b={calibration_model.region_b[i].item():.4f}")
    
    logger.info("Calibration training completed")
    return calibration_model


def evaluate_calibration(calibration_model, dataloader):
    """Evaluate calibration performance"""
    logger.info("Evaluating calibration performance")
    
    calibration_model.eval()
    
    all_base_probs = []
    all_calibrated_probs = []
    all_targets = []
    all_regions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            past_data = batch['past_data'].to(calibration_model.device)
            target = batch['target'].to(calibration_model.device)
            region_ids = batch['region_id'].to(calibration_model.device)
            
            calibrated_probs, base_probs = calibration_model(past_data, region_ids)
            
            all_base_probs.append(base_probs.cpu().numpy())
            all_calibrated_probs.append(calibrated_probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_regions.append(region_ids.cpu().numpy())
    
    # Concatenate all results
    base_probs = np.concatenate(all_base_probs, axis=0)
    calibrated_probs = np.concatenate(all_calibrated_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    regions = np.concatenate(all_regions, axis=0)
    
    # Calculate metrics
    base_nll = log_loss(targets.flatten(), base_probs.flatten())
    calibrated_nll = log_loss(targets.flatten(), calibrated_probs.flatten())
    
    base_brier = brier_score_loss(targets.flatten(), base_probs.flatten())
    calibrated_brier = brier_score_loss(targets.flatten(), calibrated_probs.flatten())
    
    logger.info("Calibration Results:")
    logger.info(f"  Base Model NLL: {base_nll:.6f}")
    logger.info(f"  Calibrated NLL: {calibrated_nll:.6f}")
    logger.info(f"  NLL Improvement: {base_nll - calibrated_nll:.6f}")
    logger.info(f"  Base Model Brier Score: {base_brier:.6f}")
    logger.info(f"  Calibrated Brier Score: {calibrated_brier:.6f}")
    logger.info(f"  Brier Improvement: {base_brier - calibrated_brier:.6f}")
    
    return {
        'base_nll': base_nll,
        'calibrated_nll': calibrated_nll,
        'base_brier': base_brier,
        'calibrated_brier': calibrated_brier,
        'base_probs': base_probs,
        'calibrated_probs': calibrated_probs,
        'targets': targets,
        'regions': regions
    }


def save_calibration_results(calibration_model, results, output_dir):
    """Save calibration results and parameters"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save calibration parameters
    params_path = os.path.join(output_dir, 'calibration_parameters.json')
    params = {
        'region_a': calibration_model.region_a.detach().cpu().numpy().tolist(),
        'region_b': calibration_model.region_b.detach().cpu().numpy().tolist(),
        'regions': calibration_model.regions
    }
    
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    logger.info(f"Calibration parameters saved to {params_path}")
    
    # Save evaluation results
    results_path = os.path.join(output_dir, 'calibration_results.json')
    eval_results = {
        'base_nll': float(results['base_nll']),
        'calibrated_nll': float(results['calibrated_nll']),
        'base_brier': float(results['base_brier']),
        'calibrated_brier': float(results['calibrated_brier']),
        'nll_improvement': float(results['base_nll'] - results['calibrated_nll']),
        'brier_improvement': float(results['base_brier'] - results['calibrated_brier'])
    }
    
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Save model state
    model_path = os.path.join(output_dir, 'calibration_model.pth')
    torch.save({
        'region_a': calibration_model.region_a.state_dict(),
        'region_b': calibration_model.region_b.state_dict(),
        'regions': calibration_model.regions,
        'num_regions': calibration_model.num_regions
    }, model_path)
    
    logger.info(f"Calibration model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Region-based Model Calibration')
    parser.add_argument('--h5-file', required=True,
                       help='Path to H5 file containing pixel data')
    parser.add_argument('--lookup-table', required=True,
                       help='Path to lookup table file (row_col region_id format)')
    parser.add_argument('--model-path', required=True,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for calibration results')
    parser.add_argument('--lookback-length', type=int, default=30,
                       help='Number of past days to use (default: 30)')
    parser.add_argument('--forecast-horizon', type=int, default=7,
                       help='Number of future days to predict (default: 7)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use (default: None for all)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base model
    base_model = load_base_model(args.model_path, args.device)
    
    # Create dataset
    dataset = RegionCalibrationDataset(
        args.h5_file, args.lookup_table, 
        args.lookback_length, args.forecast_horizon, args.max_samples
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    
    # Create calibration model
    calibration_model = RegionCalibrationModel(
        dataset.num_regions, base_model, args.device
    )
    
    # Train calibration parameters
    trained_model = train_calibration_model(
        calibration_model, dataloader, args.num_epochs, args.lr
    )
    
    # Evaluate calibration
    results = evaluate_calibration(trained_model, dataloader)
    
    # Save results
    save_calibration_results(trained_model, results, args.output_dir)
    
    logger.info("Calibration process completed successfully!")


if __name__ == '__main__':
    main()
