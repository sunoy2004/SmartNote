"""
Simple audio test to verify audio loading works
"""
import sys
sys.path.append(".")

def test_audio_loading():
    """Test audio loading from bytes"""
    try:
        from datasets import load_dataset
        print("Loading dataset...")
        # Load a single sample to test
        dataset = load_dataset("nguyenvulebinh/libris_clean_100", split="train.clean.100[:1]")
        print(f"Dataset loaded with {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            audio_data = sample['audio']
            print(f"Audio data keys: {list(audio_data.keys())}")
            
            # Test loading from bytes
            if 'bytes' in audio_data and audio_data['bytes'] is not None:
                import soundfile as sf
                import io
                import numpy as np
                
                audio_bytes = audio_data['bytes']
                print(f"Audio bytes length: {len(audio_bytes)}")
                
                # Load audio from bytes
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                print(f"Audio loaded from bytes: shape={audio_array.shape}, sr={sr}")
                
                # Convert to float32 if needed
                if audio_array.dtype != np.float32:
                    if np.issubdtype(audio_array.dtype, np.integer):
                        max_val = np.iinfo(audio_array.dtype).max
                        audio_array = audio_array.astype(np.float32) / max_val
                    else:
                        audio_array = audio_array.astype(np.float32)
                
                print("Audio loading from bytes successful!")
                return True
            else:
                print("No valid audio bytes found")
                return False
        else:
            print("Dataset is empty")
            return False
            
    except Exception as e:
        print(f"Audio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running audio test...")
    success = test_audio_loading()
    if success:
        print("✅ Audio test passed!")
    else:
        print("❌ Audio test failed!")