from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained ResNet50 for feature extraction
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Remove the classification layer to get features
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image(image_bytes: bytes):
    """
    Advanced AI image detector using multiple analysis techniques:
    1. Frequency domain analysis (FFT spectral characteristics)
    2. Color distribution analysis
    3. Gradient magnitude analysis
    4. Compression artifact detection
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    width, height = image.size
    megapixels = (width * height) / 1_000_000

    # Convert to numpy for analysis
    img_array = np.array(image)
    
    score = 0.0
    reasons = []

    # ===== 1. FREQUENCY DOMAIN ANALYSIS (FFT) =====
    # AI images often have different frequency characteristics - more uniform
    try:
        # Convert to grayscale for FFT
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        # Compute FFT with hamming window to reduce edge effects
        windowed = gray * np.hamming(gray.shape[0])[:, np.newaxis] * np.hamming(gray.shape[1])
        fft = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Normalize
        magnitude = magnitude / (np.max(magnitude) + 1e-8)
        
        # Analyze frequency patterns - compare center to outer regions
        center_h, center_w = gray.shape[0]//2, gray.shape[1]//2
        region_size = 40
        
        center_region = magnitude[max(0, center_h-region_size):min(magnitude.shape[0], center_h+region_size), 
                                   max(0, center_w-region_size):min(magnitude.shape[1], center_w+region_size)]
        
        # Get outer regions
        outer_regions = []
        if magnitude.shape[0] > 100 and magnitude.shape[1] > 100:
            outer_regions.append(magnitude[0:50, :])  # top
            outer_regions.append(magnitude[-50:, :])  # bottom
            outer_regions.append(magnitude[:, 0:50])  # left
            outer_regions.append(magnitude[:, -50:])  # right
        
        if outer_regions:
            center_mean = np.mean(center_region)
            outer_mean = np.mean(np.concatenate([r.flatten() for r in outer_regions]))
            
            # High center-to-outer ratio indicates AI smoothing
            if outer_mean > 0.001:
                freq_ratio = center_mean / (outer_mean + 0.001)
                if freq_ratio > 1.2:
                    score += 0.30
                    reasons.append('⚠️ Frequency analysis: Elevated center frequency (AI characteristic).')
                elif freq_ratio > 1.05:
                    score += 0.18
                    reasons.append('⚠️ Frequency analysis: Slightly elevated center frequency.')
                else:
                    reasons.append('✓ Frequency domain: Natural spectrum.')
            else:
                freq_ratio = 0
                reasons.append('✓ Frequency domain: Natural distribution.')
        else:
            freq_ratio = 0
            reasons.append('ℹ️ Frequency analysis: Image too small for reliable analysis.')
    except Exception as e:
        freq_ratio = 0
        reasons.append('ℹ️ Frequency analysis: Could not analyze.')

    # ===== 2. COLOR CHANNEL INDEPENDENCE ANALYSIS =====
    # Real photos have more correlated color channels; AI may show unnatural patterns
    avg_corr = 0.95  # Default to neutral
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        try:
            r_channel = img_array[:,:,0].astype(float)
            g_channel = img_array[:,:,1].astype(float)
            b_channel = img_array[:,:,2].astype(float)
            
            # Downsample for faster computation
            step = max(1, img_array.shape[0] // 64)
            r_small = r_channel[::step, ::step]
            g_small = g_channel[::step, ::step]
            b_small = b_channel[::step, ::step]
            
            r_flat = r_small.flatten()
            g_flat = g_small.flatten()
            b_flat = b_small.flatten()
            
            # Calculate correlation coefficients
            rg_corr = np.corrcoef(r_flat, g_flat)[0, 1]
            rb_corr = np.corrcoef(r_flat, b_flat)[0, 1]
            gb_corr = np.corrcoef(g_flat, b_flat)[0, 1]
            
            # Replace NaN with 0.95
            for val in [rg_corr, rb_corr, gb_corr]:
                if np.isnan(val):
                    val = 0.95
            
            avg_corr = (rg_corr + rb_corr + gb_corr) / 3
            
            # AI images often show unnatural color patterns
            if avg_corr < 0.70:
                score += 0.35
                reasons.append('⚠️ Color analysis: Very low channel correlation (strong AI indicator).')
            elif avg_corr < 0.82:
                score += 0.25
                reasons.append('⚠️ Color analysis: Low channel correlation (AI characteristic).')
            elif avg_corr < 0.88:
                score += 0.15
                reasons.append('⚠️ Color analysis: Lower than natural correlation.')
            elif avg_corr > 0.96:
                score += 0.22
                reasons.append('⚠️ Color analysis: Suspiciously perfect correlation (AI indicator).')
            else:
                reasons.append('✓ Color channels: Natural correlation pattern.')
        except:
            reasons.append('ℹ️ Color analysis: Could not analyze channels.')

    # ===== 3. GRADIENT MAGNITUDE ANALYSIS =====
    # AI images often have smoother, more uniform gradients
    try:
        gray_smooth = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        
        # Downsample for efficiency
        step = max(1, gray_smooth.shape[0] // 128)
        gray_small = gray_smooth[::step, ::step]
        
        # Calculate gradients
        grad_x = np.gradient(gray_small.astype(float), axis=1)
        grad_y = np.gradient(gray_small.astype(float), axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_std = np.std(gradient_mag)
        grad_mean = np.mean(gradient_mag)
        
        if grad_mean > 0.01:
            grad_ratio = grad_std / (grad_mean + 0.01)
            
            if grad_ratio < 0.65:  # Very smooth gradients
                score += 0.30
                reasons.append('⚠️ Edge gradients: Very smooth (strong AI indicator).')
            elif grad_ratio < 0.90:
                score += 0.22
                reasons.append('⚠️ Edge gradients: Unusually smooth (AI characteristic).')
            elif grad_ratio < 1.15:
                score += 0.12
                reasons.append('⚠️ Edge gradients: Slightly smoother than typical real photos.')
            elif grad_ratio > 2.5:
                reasons.append('✓ Edge gradients: Very natural variation.')
            else:
                reasons.append('✓ Edge gradients: Natural variation.')
        else:
            grad_ratio = 0
            reasons.append('ℹ️ Gradient analysis: Very low contrast image.')
    except:
        grad_ratio = 0
        reasons.append('ℹ️ Gradient analysis: Could not analyze.')

    # ===== 4. SATURATION AND HUE CONSISTENCY =====
    # AI images sometimes have unnatural color consistency
    try:
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Convert RGB to HSV
            from PIL import ImageOps
            hsv_image = Image.fromarray(img_array).convert('HSV')
            hsv_array = np.array(hsv_image)
            
            # Analyze saturation consistency
            if hsv_array.shape[2] >= 2:
                saturation = hsv_array[:,:,1].astype(float)
                sat_mean = np.mean(saturation)
                sat_std = np.std(saturation)
                
                if sat_mean > 0:
                    sat_ratio = sat_std / sat_mean
                    
                    if sat_ratio < 0.10:  # Too uniform saturation
                        score += 0.20
                        reasons.append('⚠️ Saturation: Extremely uniform color saturation (strong AI indicator).')
                    elif sat_ratio < 0.16:
                        score += 0.15
                        reasons.append('⚠️ Saturation: Unusually uniform color saturation (AI indicator).')
                    elif sat_ratio < 0.20:
                        score += 0.10
                        reasons.append('⚠️ Saturation: Lower than natural variation.')
                    elif sat_ratio > 0.70:
                        reasons.append('✓ Saturation: Natural variation detected.')
                    else:
                        reasons.append('ℹ️ Saturation: Moderate variation.')
    except:
        reasons.append('ℹ️ Saturation analysis: Could not analyze.')

    # ===== 5. COMPRESSION AND NOISE ANALYSIS =====
    try:
        # Laplacian variance (high = sharp, good for detecting over-smoothed images)
        if gray_smooth.size > 100:
            step = max(1, gray_smooth.shape[0] // 64)
            sample = gray_smooth[::step, ::step]
            
            # Simple laplacian
            lap = np.abs(np.gradient(np.gradient(sample, axis=0), axis=1)) + \
                  np.abs(np.gradient(np.gradient(sample, axis=1), axis=0))
            lap_mean = np.mean(lap)
            
            if lap_mean < 1.2:
                score += 0.25
                reasons.append('⚠️ Sharpness: Extremely low detail (very strong smoothing indicator).')
            elif lap_mean < 2.0:
                score += 0.20
                reasons.append('⚠️ Sharpness: Very low Laplacian variance (strong AI upscaling indicator).')
            elif lap_mean < 3.5:
                score += 0.12
                reasons.append('⚠️ Sharpness: Low detail (possible AI upscaling).')
            elif lap_mean < 5.0:
                score += 0.05
                reasons.append('⚠️ Sharpness: Slightly lower detail than expected.')
            elif lap_mean > 25.0:
                reasons.append('✓ Sharpness: High natural detail.')
            else:
                reasons.append('ℹ️ Sharpness: Natural detail level.')
    except:
        reasons.append('ℹ️ Sharpness analysis: Could not analyze.')

    # ===== 6. BASIC RESOLUTION & ASPECT CHECKS =====
    if megapixels < 0.15:
        score += 0.08
        reasons.append('⚠️ Very small resolution.')
    elif megapixels > 100:
        reasons.append('✓ Very high resolution.')

    if width > 0 and height > 0:
        aspect = max(width, height) / min(width, height)
        if aspect > 4.0:
            score += 0.05
            reasons.append('⚠️ Extreme aspect ratio.')

    # ===== FINAL SCORING =====
    # Start with baseline assumption that modern images tend to be AI
    # (since AI is increasingly prevalent)
    ai_probability = 0.35 + (score * 0.6)  # Baseline of 35% AI, then add score
    
    # Clamp between reasonable ranges
    ai_probability = min(max(ai_probability, 0.15), 0.98)
    real_probability = 1.0 - ai_probability
    
    label = 'AI-generated' if ai_probability >= 0.5 else 'Real / Human-captured'
    
    ai_percentage = round(ai_probability * 100, 2)
    real_percentage = round(real_probability * 100, 2)

    return {
        'label': label,
        'ai_probability': ai_percentage,
        'real_probability': real_percentage,
        'note': 'Advanced detector: Uses frequency analysis, color patterns, texture, and gradient analysis. Still a demo - production models require large labeled datasets.',
        'details': {
            'width': width,
            'height': height,
            'megapixels': round(megapixels, 2),
            'frequency_ratio': round(freq_ratio if 'freq_ratio' in locals() else 0, 2),
            'color_correlation': round(avg_corr, 2),
            'gradient_ratio': round(grad_ratio if 'grad_ratio' in locals() else 0, 2),
            'reasons': reasons,
        }
    }


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found. Use field name: image'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PNG, JPG, JPEG, and WEBP files are allowed.'}), 400

    image_bytes = file.read()

    try:
        result = analyze_image(image_bytes)
    except Exception as exc:
        return jsonify({'error': f'Error analyzing image: {str(exc)}'}), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
