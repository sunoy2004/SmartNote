"""
Minimal dataset test to verify the fix
"""
import sys
sys.path.append(".")

def test_minimal_dataset():
    """Test dataset with minimal configuration"""
    try:
        from datasets import load_dataset
        print("Loading dataset...")
        # Load a very small subset to test
        dataset = load_dataset("nguyenvulebinh/libris_clean_100", split="train.clean.100[:1%]")
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Check the structure of the first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            if 'audio' in sample:
                audio_data = sample['audio']
                print(f"Audio data type: {type(audio_data)}")
                if isinstance(audio_data, dict):
                    print(f"Audio data keys: {list(audio_data.keys())}")
                    # Try to load audio with librosa
                    if 'path' in audio_data:
                        import librosa
                        audio_array, sr = librosa.load(audio_data['path'], sr=None)
                        print(f"Audio loaded successfully: {audio_array.shape}, sr={sr}")
                    elif 'bytes' in audio_data:
                        import soundfile as sf
                        import io
                        audio_bytes = audio_data['bytes']
                        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                        print(f"Audio loaded from bytes: {audio_array.shape}, sr={sr}")
                else:
                    print("Audio data is not a dict")
            else:
                print("No 'audio' key in sample")
        else:
            print("Dataset is empty")
            
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running minimal dataset test...")
    success = test_minimal_dataset()
    if success:
        print("✅ Minimal dataset test passed!")
    else:
        print("❌ Minimal dataset test failed!")