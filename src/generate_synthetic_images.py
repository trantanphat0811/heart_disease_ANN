"""
Generate synthetic medical images (simulated ECG/X-ray) for CNN training.
Creates labeled image dataset for heart disease prediction.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import json
from pathlib import Path


def generate_ecg_like_image(width=224, height=224, has_disease=False, noise_level=0.1):
    """Generate synthetic ECG-like waveform image."""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw background grid
    for i in range(0, width, 20):
        draw.line([(i, 0), (i, height)], fill=(220, 220, 220), width=1)
    for i in range(0, height, 20):
        draw.line([(0, i), (width, i)], fill=(220, 220, 220), width=1)
    
    # Generate waveform
    x_points = np.linspace(0, width, 100)
    y_base = height // 2
    
    if has_disease:
        # Abnormal waveform (irregular, higher amplitude)
        y_points = y_base + 30 * np.sin(x_points / 20) + 20 * np.sin(x_points / 40) + np.random.randn(100) * noise_level * 10
    else:
        # Normal waveform (regular, lower amplitude)
        y_points = y_base + 15 * np.sin(x_points / 15) + np.random.randn(100) * noise_level * 5
    
    # Draw waveform
    points = list(zip(x_points, y_points))
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=(255, 0, 0), width=2)
    
    # Add random noise/artifacts
    for _ in range(10):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        draw.point((x, y), fill=(100, 100, 100))
    
    return np.array(img)


def generate_xray_like_image(width=224, height=224, has_disease=False, noise_level=0.1):
    """Generate synthetic X-ray-like image (heart region)."""
    # Start with background (lungs = darker)
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 150
    
    # Draw heart shape (simplified ellipse/circle)
    from PIL import Image as PILImage
    img = PILImage.new('RGB', (width, height), color=(150, 150, 150))
    draw = ImageDraw.Draw(img)
    
    # Heart center region (lighter = denser)
    heart_x, heart_y = width // 2, height // 2
    heart_size = 50 if has_disease else 40
    
    # Draw heart outline
    draw.ellipse(
        [heart_x - heart_size, heart_y - heart_size, heart_x + heart_size, heart_y + heart_size],
        fill=(100, 100, 100) if has_disease else (120, 120, 120),
        outline=(50, 50, 50)
    )
    
    # Draw vessels (normal or abnormal)
    if has_disease:
        # Abnormal vessels (irregular, blocked areas)
        draw.line([(heart_x, heart_y - heart_size), (heart_x - 30, heart_y - 80)], fill=(80, 80, 80), width=3)
        draw.line([(heart_x - 20, heart_y - heart_size), (heart_x - 50, heart_y - 40)], fill=(60, 60, 60), width=2)
    else:
        # Normal vessels (smooth, regular)
        draw.line([(heart_x, heart_y - heart_size), (heart_x - 25, heart_y - 75)], fill=(100, 100, 100), width=2)
        draw.line([(heart_x, heart_y - heart_size), (heart_x + 25, heart_y - 75)], fill=(100, 100, 100), width=2)
    
    # Add noise
    img_array = np.array(img)
    noise = np.random.normal(0, noise_level * 30, (height, width, 3))
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return img_array


def create_dataset(output_dir='data/medical_images', num_samples=1000, split=[0.7, 0.15, 0.15]):
    """
    Generate synthetic medical image dataset.
    
    Args:
        output_dir: Directory to save images
        num_samples: Total number of images to generate
        split: Train/val/test split ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/train/normal', exist_ok=True)
    os.makedirs(f'{output_dir}/train/disease', exist_ok=True)
    os.makedirs(f'{output_dir}/val/normal', exist_ok=True)
    os.makedirs(f'{output_dir}/val/disease', exist_ok=True)
    os.makedirs(f'{output_dir}/test/normal', exist_ok=True)
    os.makedirs(f'{output_dir}/test/disease', exist_ok=True)
    
    metadata = []
    
    # Calculate split counts
    train_count = int(num_samples * split[0])
    val_count = int(num_samples * split[1])
    test_count = num_samples - train_count - val_count
    
    print(f"Generating {num_samples} synthetic medical images...")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    idx = 0
    
    # Training set
    for i in range(train_count):
        has_disease = i % 2 == 0  # 50% disease, 50% normal
        img_type = np.random.choice(['ecg', 'xray'])
        
        if img_type == 'ecg':
            img = generate_ecg_like_image(has_disease=has_disease)
        else:
            img = generate_xray_like_image(has_disease=has_disease)
        
        folder = 'disease' if has_disease else 'normal'
        filename = f'image_{idx:04d}.png'
        filepath = f'{output_dir}/train/{folder}/{filename}'
        
        Image.fromarray(img).save(filepath)
        metadata.append({
            'filename': filepath,
            'split': 'train',
            'label': 1 if has_disease else 0,
            'disease': has_disease,
            'image_type': img_type
        })
        idx += 1
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1} training images...")
    
    # Validation set
    for i in range(val_count):
        has_disease = i % 2 == 0
        img_type = np.random.choice(['ecg', 'xray'])
        
        if img_type == 'ecg':
            img = generate_ecg_like_image(has_disease=has_disease)
        else:
            img = generate_xray_like_image(has_disease=has_disease)
        
        folder = 'disease' if has_disease else 'normal'
        filename = f'image_{idx:04d}.png'
        filepath = f'{output_dir}/val/{folder}/{filename}'
        
        Image.fromarray(img).save(filepath)
        metadata.append({
            'filename': filepath,
            'split': 'val',
            'label': 1 if has_disease else 0,
            'disease': has_disease,
            'image_type': img_type
        })
        idx += 1
    
    # Test set
    for i in range(test_count):
        has_disease = i % 2 == 0
        img_type = np.random.choice(['ecg', 'xray'])
        
        if img_type == 'ecg':
            img = generate_ecg_like_image(has_disease=has_disease)
        else:
            img = generate_xray_like_image(has_disease=has_disease)
        
        folder = 'disease' if has_disease else 'normal'
        filename = f'image_{idx:04d}.png'
        filepath = f'{output_dir}/test/{folder}/{filename}'
        
        Image.fromarray(img).save(filepath)
        metadata.append({
            'filename': filepath,
            'split': 'test',
            'label': 1 if has_disease else 0,
            'disease': has_disease,
            'image_type': img_type
        })
        idx += 1
    
    # Save metadata
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dataset created at {output_dir}")
    print(f"   Total images: {len(metadata)}")
    print(f"   Metadata saved: {output_dir}/metadata.json")
    
    return output_dir


if __name__ == '__main__':
    create_dataset('data/medical_images', num_samples=1000)
