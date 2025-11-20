"""
Flask Backend API for Custom ASR and NLP Models
Provides endpoints for transcription and summarization
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import tempfile
from werkzeug.utils import secure_filename

# Import custom models
import sys
sys.path.append('asr')
sys.path.append('nlp')

from asr.model import create_model as create_asr_model
from asr.utils_audio import AudioProcessor
from asr.ctc_decoder import decode_predictions

from nlp.model_summarizer import create_summarizer
from nlp.tokenizer import SimpleTokenizer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Model paths
ASR_CHECKPOINT = 'asr/checkpoints/best_model.pth'
NLP_CHECKPOINT = 'nlp/checkpoints/best_summarizer.pth'

# Global model instances (loaded once at startup)
asr_model = None
nlp_model = None
audio_processor = None
tokenizer = None
device = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load models at startup"""
    global asr_model, nlp_model, audio_processor, tokenizer, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ASR model
    if os.path.exists(ASR_CHECKPOINT):
        print("Loading ASR model...")
        asr_model = create_asr_model().to(device)
        checkpoint = torch.load(ASR_CHECKPOINT, map_location=device)
        asr_model.load_state_dict(checkpoint['model_state_dict'])
        asr_model.eval()
        audio_processor = AudioProcessor()
        print("✓ ASR model loaded")
    else:
        print(f"⚠ ASR checkpoint not found at {ASR_CHECKPOINT}")
    
    # Load NLP model
    if os.path.exists(NLP_CHECKPOINT):
        print("Loading NLP model...")
        tokenizer = SimpleTokenizer()
        nlp_model = create_summarizer(tokenizer.vocab_size).to(device)
        checkpoint = torch.load(NLP_CHECKPOINT, map_location=device)
        nlp_model.load_state_dict(checkpoint['model_state_dict'])
        nlp_model.eval()
        print("✓ NLP model loaded")
    else:
        print(f"⚠ NLP checkpoint not found at {NLP_CHECKPOINT}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'asr_loaded': asr_model is not None,
        'nlp_loaded': nlp_model is not None
    })

@app.route('/api/asr/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file using custom ASR model
    
    Expects:
        - 'audio' file in form data
    
    Returns:
        - {'transcript': 'transcribed text'}
    """
    if asr_model is None:
        return jsonify({'error': 'ASR model not loaded'}), 503
    
    # Check file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        # Process audio
        mel_spec = audio_processor.process_audio(tmp_path)
        mel_spec = mel_spec.to(device)
        
        # Transcribe
        with torch.no_grad():
            log_probs = asr_model(mel_spec)
            transcripts = decode_predictions(log_probs)
        
        transcript = transcripts[0] if transcripts else ''
        
        # Cleanup
        os.remove(tmp_path)
        
        return jsonify({'transcript': transcript})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nlp/summarize', methods=['POST'])
def summarize():
    """
    Summarize text using custom NLP model
    
    Expects:
        - {'text': 'text to summarize'}
    
    Returns:
        - {'summary': 'summarized text'}
    """
    if nlp_model is None:
        return jsonify({'error': 'NLP model not loaded'}), 503
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Empty text'}), 400
    
    try:
        # Encode text
        src_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # Pad to max length
        max_len = 512
        if len(src_ids) > max_len:
            src_ids = src_ids[:max_len-1] + [tokenizer.vocab[tokenizer.EOS_TOKEN]]
        src_ids += [tokenizer.vocab[tokenizer.PAD_TOKEN]] * (max_len - len(src_ids))
        
        # Convert to tensor
        src_tensor = torch.LongTensor([src_ids]).to(device)
        
        # Generate summary
        with torch.no_grad():
            generated = nlp_model.generate(
                src_tensor,
                max_length=100,
                start_token=tokenizer.vocab[tokenizer.BOS_TOKEN],
                end_token=tokenizer.vocab[tokenizer.EOS_TOKEN]
            )
        
        # Decode
        summary = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
        
        return jsonify({'summary': summary})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice-to-notes', methods=['POST'])
def voice_to_notes():
    """
    Complete pipeline: audio -> transcript -> summary
    
    Expects:
        - 'audio' file in form data
    
    Returns:
        - {'transcript': '...', 'summary': '...'}
    """
    if asr_model is None or nlp_model is None:
        return jsonify({'error': 'Models not loaded'}), 503
    
    # Check file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid audio file'}), 400
    
    try:
        # Step 1: Transcribe
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        mel_spec = audio_processor.process_audio(tmp_path)
        mel_spec = mel_spec.to(device)
        
        with torch.no_grad():
            log_probs = asr_model(mel_spec)
            transcripts = decode_predictions(log_probs)
        
        transcript = transcripts[0] if transcripts else ''
        os.remove(tmp_path)
        
        if not transcript.strip():
            return jsonify({'error': 'Empty transcription'}), 400
        
        # Step 2: Summarize
        src_ids = tokenizer.encode(transcript, add_bos=True, add_eos=True)
        max_len = 512
        if len(src_ids) > max_len:
            src_ids = src_ids[:max_len-1] + [tokenizer.vocab[tokenizer.EOS_TOKEN]]
        src_ids += [tokenizer.vocab[tokenizer.PAD_TOKEN]] * (max_len - len(src_ids))
        
        src_tensor = torch.LongTensor([src_ids]).to(device)
        
        with torch.no_grad():
            generated = nlp_model.generate(
                src_tensor,
                max_length=100,
                start_token=tokenizer.vocab[tokenizer.BOS_TOKEN],
                end_token=tokenizer.vocab[tokenizer.EOS_TOKEN]
            )
        
        summary = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
        
        return jsonify({
            'transcript': transcript,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("Custom ASR + NLP Backend API")
    print("="*60)
    load_models()
    print("\nStarting server on http://0.0.0.0:5000")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /api/asr/transcribe")
    print("  POST /api/nlp/summarize")
    print("  POST /api/voice-to-notes")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)
