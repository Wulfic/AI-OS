# Launchpad PPA Setup Guide

This guide explains how to configure the GitHub Actions workflow to publish AI-OS packages to a Launchpad PPA.

## Required GitHub Secrets

You need to configure two secrets in your GitHub repository:

### 1. LAUNCHPAD_GPG_PRIVATE_KEY

Your GPG private key in ASCII-armored format.

**To get your private key:**
```bash
gpg --armor --export-secret-keys YOUR_KEY_ID
```

**Copy the entire output including:**
- `-----BEGIN PGP PRIVATE KEY BLOCK-----`
- The key content
- `-----END PGP PRIVATE KEY BLOCK-----`

### 2. LAUNCHPAD_GPG_PASSPHRASE

The passphrase you used when creating your GPG key.

## Setting Up GitHub Secrets

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret:
   - Name: `LAUNCHPAD_GPG_PRIVATE_KEY`
   - Value: Your private key (paste the entire block)
   - Click **Add secret**
5. Repeat for `LAUNCHPAD_GPG_PASSPHRASE`

## GPG Key Requirements

Your GPG key must be:
- ✓ Registered with Launchpad (upload your public key to https://launchpad.net/~/+editpgpkeys)
- ✓ Associated with the same email used in your Launchpad account
- ✓ Not expired
- ✓ At least 2048-bit RSA (4096-bit recommended)

## Configuring Your PPA

Edit the workflow file or use workflow dispatch with your PPA name:
```
ppa:YOUR_USERNAME/YOUR_PPA_NAME
```

Example: `ppa:wulfic/ai-os`

## Privacy and Security

The workflow is designed with privacy in mind:
- ✓ Secrets are never logged or exposed
- ✓ GPG operations use `--quiet` and `--batch` flags
- ✓ Sensitive outputs are masked with `::add-mask::`
- ✓ Only the last 8 characters of key IDs are shown in logs
- ✓ Generic maintainer info is used in package metadata (`AI-OS Team <contact@example.com>`)

## Testing the Workflow

1. **Manual trigger**: Go to **Actions** → **Publish to Launchpad PPA** → **Run workflow**
2. **Automatic trigger**: The workflow runs after successful installer tests

## Troubleshooting

### GPG Key Not Found
- Verify secrets are properly configured in GitHub
- Check that the key hasn't expired
- Ensure the key is properly formatted (includes BEGIN/END markers)

### Upload Rejected by Launchpad
- Confirm your GPG key is registered with Launchpad
- Verify the email in your GPG key matches your Launchpad account
- Check that the package version doesn't already exist in your PPA

### Build Failures
- Review Launchpad build logs at `https://launchpad.net/~YOUR_USERNAME/+archive/ubuntu/YOUR_PPA/+packages`
- Verify debian packaging is correct
- Check dependencies are available in Ubuntu repositories

## Additional Resources

- [Launchpad PPA Guide](https://help.launchpad.net/Packaging/PPA)
- [GPG Key Management](https://help.ubuntu.com/community/GnuPrivacyGuardHowto)
- [Debian Packaging](https://www.debian.org/doc/manuals/maint-guide/)
