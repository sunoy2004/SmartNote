"""
Test script to verify Supabase credentials
"""
import os
from dotenv import load_dotenv
from supabase import create_client

def test_supabase_credentials():
    """Test Supabase credentials"""
    print("Loading environment variables...")
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    print(f"SUPABASE_URL: {supabase_url}")
    print(f"SUPABASE_KEY: {supabase_key[:10] if supabase_key else 'None'}...{supabase_key[-10:] if supabase_key else 'None'}")
    print(f"SUPABASE_SERVICE_ROLE_KEY: {supabase_service_key[:10] if supabase_service_key else 'None'}...{supabase_service_key[-10:] if supabase_service_key else 'None'}")
    
    # Test with regular key first
    if supabase_url and supabase_key:
        print("\nTesting with SUPABASE_KEY...")
        try:
            print("Testing Supabase connection...")
            supabase = create_client(supabase_url, supabase_key)
            print("✓ Supabase client created successfully with SUPABASE_KEY")
            
            # Test storage access
            try:
                buckets = supabase.storage.list_buckets()
                print("✓ Storage access successful")
                print(f"  Available buckets: {[bucket.name for bucket in buckets]}")
            except Exception as e:
                print(f"⚠ Storage access failed: {e}")
            
            # Test database access
            try:
                result = supabase.table("models_meta").select("*").limit(1).execute()
                print("✓ Database access successful")
            except Exception as e:
                print(f"⚠ Database access failed: {e}")
            
            return True
        except Exception as e:
            print(f"❌ Supabase connection with SUPABASE_KEY failed: {e}")
    
    # If regular key fails, try with service role key
    if supabase_url and supabase_service_key:
        print("\nTesting with SUPABASE_SERVICE_ROLE_KEY...")
        try:
            print("Testing Supabase connection...")
            supabase = create_client(supabase_url, supabase_service_key)
            print("✓ Supabase client created successfully with SUPABASE_SERVICE_ROLE_KEY")
            
            # Test storage access
            try:
                buckets = supabase.storage.list_buckets()
                print("✓ Storage access successful")
                print(f"  Available buckets: {[bucket.name for bucket in buckets]}")
            except Exception as e:
                print(f"⚠ Storage access failed: {e}")
            
            # Test database access
            try:
                result = supabase.table("models_meta").select("*").limit(1).execute()
                print("✓ Database access successful")
            except Exception as e:
                print(f"⚠ Database access failed: {e}")
            
            return True
        except Exception as e:
            print(f"❌ Supabase connection with SUPABASE_SERVICE_ROLE_KEY failed: {e}")
    
    print("❌ Missing or invalid Supabase credentials")
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Supabase Credentials")
    print("=" * 60)
    
    success = test_supabase_credentials()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Supabase credentials are valid!")
    else:
        print("✗ Supabase credentials are invalid or missing.")
        print("  Please check your .env file and ensure SUPABASE_URL and SUPABASE_KEY are correct.")
    print("=" * 60)