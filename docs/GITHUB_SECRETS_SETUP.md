# Quick Setup: GitHub Secrets for Launchpad PPA

## Step 1: Navigate to Repository Secrets

1. Go to https://github.com/Wulfic/AI-OS
2. Click **Settings** (top menu)
3. Scroll down to **Secrets and variables** → **Actions**
4. Click **New repository secret**

## Step 2: Add LAUNCHPAD_GPG_PRIVATE_KEY

**Name:** `LAUNCHPAD_GPG_PRIVATE_KEY`

**Value:** Paste your ENTIRE private key including:
```
-----BEGIN PGP PRIVATE KEY BLOCK-----
[Your key content here]
-----END PGP PRIVATE KEY BLOCK-----
```

**Important:**
- Include ALL lines from BEGIN to END
- Do not add extra spaces or line breaks
- Make sure it's the PRIVATE key, not public

Click **Add secret**

## Step 3: Add LAUNCHPAD_GPG_PASSPHRASE

**Name:** `LAUNCHPAD_GPG_PASSPHRASE`

**Value:** Your GPG key passphrase (example: `Sudo666rm*.*`)

**Important:**
- Type or paste exactly as it is
- Do not include quotes
- Case-sensitive

Click **Add secret**

## Step 4: Verify Secrets

After adding both secrets, you should see:
- ✅ LAUNCHPAD_GPG_PRIVATE_KEY (Updated X seconds ago)
- ✅ LAUNCHPAD_GPG_PASSPHRASE (Updated X seconds ago)

## Step 5: Upload Public Key to Launchpad

1. Go to https://launchpad.net/~/+editpgpkeys
2. Paste your PUBLIC key (from the file you have)
3. Click "Import Key"
4. Check your email for confirmation
5. Click the confirmation link

## Step 6: Test the Workflow

1. Go to **Actions** tab in GitHub
2. Select **Publish to Launchpad PPA** workflow
3. Click **Run workflow**
4. Enter your PPA name: `ppa:wulfic/ai-os`
5. Click **Run workflow**

## Troubleshooting

### Secret Not Working?
- Check for extra spaces or newlines
- Verify the key format (BEGIN/END markers)
- Make sure passphrase is typed correctly

### Upload Fails?
- Confirm public key is uploaded to Launchpad
- Check that email in GPG key matches Launchpad account
- Verify PPA exists and you have access

### Need to Update Secrets?
1. Go back to repository secrets
2. Click on the secret name
3. Click **Update secret**
4. Paste new value
5. Click **Update secret**

---

**Security Note:** These secrets are encrypted and never exposed in logs. The workflow uses secure masking to protect sensitive data.
