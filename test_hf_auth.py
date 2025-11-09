#!/usr/bin/env python3
"""
HuggingFace Authentication Diagnostic Tool

This script tests various aspects of HuggingFace authentication to identify
why read-only tokens might be failing verification.
"""

import sys
from pathlib import Path

print("="*80)
print("🔍 HuggingFace Authentication Diagnostic Tool")
print("="*80)

# Check if huggingface_hub is installed
try:
    from huggingface_hub import login, whoami, HfFolder, HfApi
    try:
        from huggingface_hub.errors import HfHubHTTPError  # type: ignore
    except ImportError:
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore
    print("✅ huggingface_hub library is installed")
except ImportError as e:
    print(f"❌ huggingface_hub library is NOT installed: {e}")
    print("\nInstalling huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import login, whoami, HfFolder, HfApi
    try:
        from huggingface_hub.errors import HfHubHTTPError  # type: ignore
    except ImportError:
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore
    print("✅ huggingface_hub library installed successfully")

print("\n" + "="*80)
print("📋 Current Authentication Status")
print("="*80)

# Check for existing token
try:
    token = HfFolder.get_token()
    if token:
        print(f"✅ Token found in HfFolder")
        print(f"   Token (first 10 chars): {token[:10]}...")
        print(f"   Token length: {len(token)} characters")
    else:
        print("❌ No token found in HfFolder")
except Exception as e:
    print(f"❌ Error getting token from HfFolder: {e}")
    token = None

# Test whoami() with existing token
if token:
    print("\n" + "-"*80)
    print("Testing whoami() with existing token...")
    print("-"*80)
    try:
        user_info = whoami(token=token)
        print(f"✅ whoami() succeeded!")
        print(f"   User info type: {type(user_info)}")
        print(f"   User info keys: {list(user_info.keys()) if isinstance(user_info, dict) else 'N/A'}")
        
        if isinstance(user_info, dict):
            username = user_info.get('name', user_info.get('id', 'Unknown'))
            email = user_info.get('email', 'N/A')
            fullname = user_info.get('fullname', 'N/A')
            print(f"   Username: {username}")
            print(f"   Email: {email}")
            print(f"   Full name: {fullname}")
        else:
            print(f"   User info: {user_info}")
    except HfHubHTTPError as e:
        print(f"❌ whoami() failed with HTTP error:")
        print(f"   Status code: {e.response.status_code}")
        print(f"   Message: {e.response.text[:200]}")
    except Exception as e:
        print(f"❌ whoami() failed: {type(e).__name__}: {e}")

# Test HfApi
print("\n" + "-"*80)
print("Testing HfApi.whoami()...")
print("-"*80)
try:
    api = HfApi()
    if token:
        api_user = api.whoami(token=token)
        print(f"✅ HfApi.whoami() succeeded!")
        print(f"   Result type: {type(api_user)}")
        print(f"   Result: {api_user}")
    else:
        print("⚠️  No token to test with")
except Exception as e:
    print(f"❌ HfApi.whoami() failed: {type(e).__name__}: {e}")

# Prompt for manual token test
print("\n" + "="*80)
print("🧪 Manual Token Test")
print("="*80)
print("\nYou can test your read-only token here.")
print("The token will NOT be saved - this is just for testing.\n")

import getpass
test_token = getpass.getpass("Enter your HuggingFace token (or press Enter to skip): ").strip()

if test_token:
    print("\n" + "-"*80)
    print("Testing provided token...")
    print("-"*80)
    print(f"Token (first 10 chars): {test_token[:10]}...")
    print(f"Token length: {len(test_token)} characters")
    
    # Test 1: whoami()
    print("\n1️⃣ Testing whoami()...")
    try:
        user_info = whoami(token=test_token)
        print(f"   ✅ whoami() succeeded!")
        print(f"   Type: {type(user_info)}")
        
        if isinstance(user_info, dict):
            username = user_info.get('name', user_info.get('id', 'Unknown'))
            print(f"   Username: {username}")
            print(f"   Full response: {user_info}")
        else:
            print(f"   Response: {user_info}")
    except HfHubHTTPError as e:
        print(f"   ❌ HTTP Error {e.response.status_code}")
        print(f"   Message: {e.response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Failed: {type(e).__name__}: {e}")
    
    # Test 2: HfApi.whoami()
    print("\n2️⃣ Testing HfApi.whoami()...")
    try:
        api = HfApi()
        api_user = api.whoami(token=test_token)
        print(f"   ✅ HfApi.whoami() succeeded!")
        print(f"   Type: {type(api_user)}")
        print(f"   Response: {api_user}")
    except Exception as e:
        print(f"   ❌ Failed: {type(e).__name__}: {e}")
    
    # Test 3: login()
    print("\n3️⃣ Testing login() (will save token)...")
    save_choice = input("   Do you want to save this token? (y/n): ").strip().lower()
    
    if save_choice == 'y':
        try:
            login_result = login(token=test_token, add_to_git_credential=True)
            print(f"   ✅ login() succeeded!")
            print(f"   Result: {login_result}")
            
            # Verify token was saved
            saved_token = HfFolder.get_token()
            if saved_token:
                print(f"   ✅ Token saved successfully")
                
                # Test whoami with saved token
                try:
                    user_info = whoami(token=saved_token)
                    username = user_info.get('name', user_info.get('id', 'Unknown')) if isinstance(user_info, dict) else 'Unknown'
                    print(f"   ✅ Verified - logged in as: {username}")
                except:
                    print(f"   ⚠️  Token saved but verification failed")
            else:
                print(f"   ⚠️  login() succeeded but token not found in HfFolder")
                
        except Exception as e:
            print(f"   ❌ login() failed: {type(e).__name__}: {e}")
    else:
        print("   ⏭️  Skipped saving token")
    
    # Test 4: Try accessing a gated dataset
    print("\n4️⃣ Testing access to gated dataset (bigcode/the-stack)...")
    try:
        from datasets import load_dataset
        print("   Attempting to load dataset info...")
        
        # This should work with read-only token if user has access
        dataset = load_dataset("bigcode/the-stack", "python", split="train", streaming=True, token=test_token)
        print("   ✅ Dataset access succeeded!")
        print("   ✅ Your read-only token CAN access gated datasets!")
        
        # Try getting one sample
        try:
            sample = next(iter(dataset))
            if isinstance(sample, dict):
                print(f"   ✅ Successfully retrieved sample: {list(sample.keys())[:5]}...")
            else:
                print(f"   ✅ Successfully retrieved sample (type: {type(sample)})")
        except Exception as e:
            print(f"   ⚠️  Could not retrieve sample: {e}")
            
    except Exception as e:
        print(f"   ❌ Dataset access failed: {type(e).__name__}")
        print(f"   Error: {str(e)[:300]}")
        print("\n   This could mean:")
        print("   - You haven't accepted the dataset's terms of use")
        print("   - Your token doesn't have the right permissions")
        print("   - The dataset requires additional authentication")
else:
    print("⏭️  Skipped manual token test")

# Summary and recommendations
print("\n" + "="*80)
print("📊 Summary & Recommendations")
print("="*80)

print("\n🔍 Common Issues with Read-Only Tokens:")
print("   1. Token not yet validated - may take a few minutes after creation")
print("   2. Haven't accepted dataset terms (visit dataset page and click 'Agree')")
print("   3. Token has wrong scope - needs 'read' permission")
print("   4. API might return unexpected format - our code expects dict with 'name'")

print("\n💡 Debugging Steps:")
print("   1. Visit: https://huggingface.co/settings/tokens")
print("   2. Verify your token shows 'read' permission")
print("   3. For gated datasets, visit their page and click 'Agree and access'")
print("   4. Wait 1-2 minutes after creating token")
print("   5. Try token in this script to see exact error")

print("\n🛠️  Code Improvements Needed:")
print("   1. Better error messages (show exact API response)")
print("   2. Handle different whoami() response formats")
print("   3. Check for 'id' field if 'name' is missing")
print("   4. Add retry logic for newly created tokens")

print("\n" + "="*80)
print("✅ Diagnostic Complete!")
print("="*80)
